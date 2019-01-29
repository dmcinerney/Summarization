import torch
from torch import nn
from torch.nn import functional as F
from beam_search import beam_search
from submodules import LSTMTextEncoder, LSTMSummaryDecoder, ContextVectorNN, VocabularyDistributionNN, ProbabilityNN
from model_helpers import GeneratedSummary, GeneratedSummaryHypothesis, PointerInfo, trim_text
import parameters as p
import pdb

# Outline:
# a) Summarizer
# b) Encoder
# c) Decoder

class Summarizer(nn.Module):
    def __init__(self, vectorizer, start_index, end_index, lstm_hidden=None, attn_hidden=None, with_coverage=False, gamma=1, with_pointer=False, encoder_base=LSTMTextEncoder, decoder_base=LSTMSummaryDecoder, decoder_parallel_base=None):
        super(Summarizer, self).__init__()

        self.vectorizer = vectorizer
        self.start_index = start_index
        self.end_index = end_index
        self.lstm_hidden = len(vectorizer.word_vectors[0])//2 if lstm_hidden is None else lstm_hidden
        self.attn_hidden = attn_hidden
        self.with_coverage = with_coverage
        self.gamma = gamma
        self.with_pointer = with_pointer
        self.encoder_base = encoder_base
        self.decoder_base = decoder_base
        self.decoder_parallel_base = decoder_parallel_base
        self.init_submodules()

    def init_submodules(self):
        decoder_class = Decoder if not self.with_pointer else PointerGenDecoder
        self.encoder = Encoder(self.vectorizer, self.lstm_hidden, encoder_base=self.encoder_base)
        self.decoder = decoder_class(self.vectorizer, self.start_index, self.end_index, self.lstm_hidden, attn_hidden=self.attn_hidden, with_coverage=self.with_coverage, gamma=self.gamma, decoder_base=self.decoder_base, decoder_parallel_base=self.decoder_parallel_base)

    def forward(self, text, text_length, text_oov_indices=None, summary=None, summary_length=None, beam_size=1):
        text, text_length = trim_text(text, text_length, p.MAX_TEXT_LENGTH)
        if summary is not None:
            summary, summary_length = trim_text(summary, summary_length, p.MAX_SUMMARY_LENGTH)
        text_states, state = self.encoder(text, text_length)
        if self.with_pointer:
            self.decoder.set_pointer_info(PointerInfo(text, text_oov_indices))
        return self.decoder(text_states, text_length, state, summary=summary, summary_length=summary_length, beam_size=beam_size)

class Encoder(nn.Module):
    def __init__(self, vectorizer, lstm_hidden, encoder_base=LSTMTextEncoder):
        super(Encoder, self).__init__()
        self.vectorizer = vectorizer
        num_features = len(self.vectorizer.word_vectors[0])

        self.text_encoder = encoder_base(num_features, lstm_hidden)

    def forward(self, text, text_length):
        # get batch with vectors from index batch
#         text = [self.vectorizer.get_text_matrix(example[:text_length[i]], text.size(1))[0].unsqueeze(0) for i,example in enumerate(text)]
#         text = torch.cat(text, 0)
        text = self.vectorizer(text, text_length)

        # run text through lstm encoder
        text_states, state = self.text_encoder(text, text_length)

        return text_states, state

class Decoder(nn.Module):
    def __init__(self, vectorizer, start_index, end_index, lstm_hidden, attn_hidden=None, with_coverage=False, gamma=1, decoder_base=LSTMSummaryDecoder, decoder_parallel_base=None):
        super(Decoder, self).__init__()
        self.vectorizer = vectorizer
        self.start_index = start_index
        self.end_index = end_index
        self.lstm_hidden = lstm_hidden
        self.attn_hidden = num_features//2 if attn_hidden is None else attn_hidden
        self.with_coverage = with_coverage
        self.gamma = gamma

        self.num_features = len(self.vectorizer.word_vectors[0])
        self.num_vocab = len(self.vectorizer.word_vectors)
        self.decoder_base = decoder_base
        self.decoder_parallel_base = decoder_parallel_base
        self.init_submodules()

    def init_submodules(self):
        self.summary_decoder = self.decoder_base(self.num_features, self.lstm_hidden)
        if self.decoder_parallel_base is not None:
            self.summary_decoder_parallel = self.decoder_parallel_base(self.summary_decoder)
        self.context_nn = ContextVectorNN(self.lstm_hidden*4+1, self.attn_hidden)
        self.vocab_nn = VocabularyDistributionNN(self.lstm_hidden*4, self.lstm_hidden, self.num_vocab+1)

    def forward(self, text_states, text_length, state, summary=None, summary_length=None, beam_size=1):
        if summary is None:
            return self.decode_generate(text_states, text_length, state, beam_size=beam_size)
        else:
            return self.decode_train(text_states, text_length, state, summary, summary_length)
#             return self.decode_train_optimized(text_states, text_length, state, summary, summary_length)

    def decode_generate(self, text_states, text_length, state, beam_size=1):
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        generated_summary = GeneratedSummary(batch_length, device, self.start_index, self.end_index)
        coverage = torch.zeros((batch_length, text_states.size(1)), device=device)
        hypothesis = GeneratedSummaryHypothesis(self, generated_summary, text_states, text_length, state, coverage)

        summary_hyps = beam_search(hypothesis.next_hypotheses(beam_size), beam_size)
        results = [summary_hyp.generated_summary.return_info() for summary_hyp in summary_hyps]

        for r in results:
            indices = r[0]
            self.map_generated_indices_(indices)

        return results

    # implements the forward pass of the decoder for training
    # this uses teacher forcing, but conceivably one could try
    # and should add other algorithms for training
    def decode_train(self, text_states, text_length, state_t, summary, summary_length):
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        coverage_t = torch.zeros((batch_length, text_states.size(1)), device=device)
        loss_unnormalized = torch.zeros(batch_length, device=device)
        summary_tp1 = summary[:,0]
        for t in range(summary.size(1)-1):
            # set timestep words
            summary_t = summary_tp1

            # get indices of instances that are not finished
            mask_t = (summary_length-1-t) > 0

            # take a time step
            vocab_dist_t, output_t, state_temp, attention_t, context_vector_t = self.timestep(mask_t, summary_t[mask_t], text_states[mask_t], text_length[mask_t], state_t[mask_t] if state_t is not None else None, coverage_t[mask_t])
            state_t = torch.zeros((batch_length,*state_temp.shape[1:]), device=state_temp.device)
            state_t[mask_t] = state_temp

            # get next time step words
            summary_tp1 = summary[:,t+1]

            # calculate log prob, calculate covloss if aplicable, update coverage if aplicable
            self.map_input_indices_(summary_tp1)
            log_prob_t = self.calculate_log_prob(vocab_dist_t, summary_tp1[mask_t])
            if self.with_coverage:
                covloss_t = self.calculate_covloss(coverage_t[mask_t], attention_t)
                coverage_t[mask_t] += attention_t

            # update unnormalized loss
            loss_unnormalized[mask_t] += (-log_prob_t + self.gamma*(covloss_t if self.with_coverage else 0))

        return dict(loss=(loss_unnormalized/(summary_length.float()-1)))

#     def decode_train_optimized(self, text_states, text_length, state, summary, summary_length):
#         outputs, context_vectors, attentions, coverages = self.parallelized_pass(text_states, text_length, state, summary[:,:-1])
#         targets = summary[:,1:]
#         b, s_d = targets.size()
#         losses = []
#         for t in range(s_d):
#             vocab_dist = self.vocab_nn(context_vectors[:,t], outputs[:,t])
#             target = targets[:,t]
#             self.map_input_indices_(target)
#             log_prob = self.calculate_log_prob(vocab_dist, target)
#             loss = -log_prob + self.gamma*(self.calculate_covloss(coverages[:,t], attentions[:,t]) if self.with_coverage else 0)
#             losses.append(loss.unsqueeze(1))
#         losses = torch.cat(losses, 1)
#         mask = torch.arange(s_d, device=summary_length.device, dtype=torch.long).unsqueeze(0) < (summary_length.unsqueeze(1)-1)
#         loss_unnormalized = (losses*mask.float()).sum(1)
#         return dict(loss=(loss_unnormalized/(summary_length.float()-1)))

#     def parallelized_pass(self, text_states, text_length, state, summary):
#         # pass through the vectorizer
#         s_e = text_states.size(1)
#         b, s_d = summary.size()
#         summary = self.vectorizer(summary, torch.ones(b, dtype=torch.long)*s_d)

#         # pass through the decoder base (can only be parallelized when given a decoder parallel base)
#         if self.decoder_parallel_base is not None:
#             outputs, state = self.summary_decoder_parallel(summary, state)
#         else:
#             outputs = []
#             for t in range(s_d):
#                 output, state = self.summary_decoder(summary[:,t], state)
#                 outputs.append(output.unsqueeze(1))
#             outputs = torch.cat(outputs, 1)

#         # pass through the attention (can only be parallelized when not with coverage)
#         coverage = torch.zeros((b, s_e), device=summary.device)
#         if not self.with_coverage:
#             context_vectors, attentions = self.context_nn(text_states, text_length, outputs, coverage)
#             coverages = coverage.unsqueeze(1).expand(b, s_d, s_e)
#         else:
#             context_vectors, attentions, coverages = [], [], []
#             for t in range(s_d):
#                 context_vector, attention = self.context_nn(text_states, text_length, outputs[:,t].unsqueeze(1), coverage)
#                 context_vectors.append(context_vector)
#                 attentions.append(attention)
#                 coverages.append(coverage.unsqueeze(1))
#                 coverage = coverage + attention.squeeze(1)
#             context_vectors, attentions, coverages = torch.cat(context_vectors, 1), torch.cat(attentions, 1), torch.cat(coverages, 1)

#         return outputs, context_vectors, attentions, coverages

    # runs the inputs for a time step through the neural nets to get the vocab distribution for that timestep
    # and other necessary information: inputs to the next hidden state in the decoder, attention, and the context vector
    # (the context vector is only needed in the subclass of this so kinda bad style but whatever)
    def timestep(self, mask_t, summary_t, text_states, text_length, state_t, coverage_t):
        summary_vec_t = self.vectorizer.get_text_matrix(summary_t, len(summary_t))[0]

        output_t, state_t = self.summary_decoder(summary_vec_t, state_t)
        context_vector_t, attention_t = self.context_nn(text_states, text_length, output_t.unsqueeze(1), coverage_t)
        context_vector_t, attention_t = context_vector_t[:,0], attention_t[:,0]
        vocab_dist_t = self.vocab_nn(context_vector_t, output_t)
        return vocab_dist_t, output_t, state_t, attention_t, context_vector_t

    # calulates the log probability of the summary at a time step given the vocab distribution for that time step
    def calculate_log_prob(self, vocab_dist, summary_tp1):
        return torch.log(vocab_dist[torch.arange(summary_tp1.size(0)).long(),summary_tp1.long()])

    # calculates the coverage loss for each batch example at a time step
    def calculate_covloss(self, coverage, attention):
        return torch.min(torch.cat((coverage.unsqueeze(0), attention.unsqueeze(0)), 0), 0)[0].sum(1)

    # map oov indices maps the indices of oov words to a specific index corresponding to the position in the vocab distribution
    # that represents an oov word
    def map_input_indices_(self, indices):
        indices[indices == -1] = len(self.vectorizer.word_vectors)

    def map_generated_indices_(self, indices):
        indices[indices == len(self.vectorizer.word_vectors)] = -1

    # this adds any extra information you may want to add to a summary
    def get_extras(self):
        return ()

# This model subclasses the generator model so that on each forward timestep, it averages the generator vocab distribution
# with a pointer distribution obtained from the attention distribution and these are weighted by p_gen and 1-p_gen respectively
# where p_gen is the probability of generating vs copying
class PointerGenDecoder(Decoder):
    def __init__(self, *args, **kwargs):
        super(PointerGenDecoder, self).__init__(*args, **kwargs)
        self.pointer_info = None

    def init_submodules(self):
        super(PointerGenDecoder, self).init_submodules()
        self.probability_layer = ProbabilityNN(self.lstm_hidden*4)

    def set_pointer_info(self, pointer_info):
        self.pointer_info = pointer_info

    # this is a little bit of a hacky solution, setting the pointer info as an object attribute temporarily
    def forward(self, *args, **kwargs):
        return_values = super(PointerGenDecoder, self).forward(*args, **kwargs)
        self.pointer_info = None
        return return_values

#     def decode_train_optimized(self, text_states, text_length, state, summary, summary_length):
#         outputs, context_vectors, attentions, coverages = self.parallelized_pass(text_states, text_length, state, summary[:,:-1])
#         targets = summary[:,1:]
#         b, s_d = targets.size()
#         losses = []
#         for t in range(s_d):
#             vocab_dist = self.vocab_nn(context_vectors[:,t], outputs[:,t])
#             final_vocab_dist = self.timestep_addon(vocab_dist, outputs[:,t], attentions[:,t], context_vectors[:,t])
#             target = targets[:,t]
#             self.map_input_indices_(target)
#             log_prob = self.calculate_log_prob(final_vocab_dist, target)
#             loss = -log_prob + self.gamma*(self.calculate_covloss(coverages[:,t], attentions[:,t]) if self.with_coverage else 0)
#             losses.append(loss.unsqueeze(1))
#         losses = torch.cat(losses, 1)
#         mask = torch.arange(s_d, device=summary_length.device, dtype=torch.long).unsqueeze(0) < (summary_length.unsqueeze(1)-1)
#         loss_unnormalized = (losses*mask.float()).sum(1)
#         return dict(loss=(loss_unnormalized/(summary_length.float()-1)))

    def timestep(self, mask_t, summary_t, text_states, text_length, state_t, coverage_t):
        if self.pointer_info is None:
            raise Exception

        # execute the normal timestep_forward function
        vocab_dist_t, output_t, state_t, attention_t, context_vector_t = super(PointerGenDecoder, self).timestep(mask_t, summary_t, text_states, text_length, state_t, coverage_t)

        final_vocab_dist_t = self.timestep_addon(mask_t, vocab_dist_t, output_t, attention_t, context_vector_t)

        return final_vocab_dist_t, output_t, state_t, attention_t, context_vector_t

    def timestep_addon(self, mask_t, vocab_dist_t, output_t, attention_t, context_vector_t):
        # get probability of generating vs copying
        p_gen_t = self.probability_layer(context_vector_t, output_t)
        self.pointer_info.update_p_gen(p_gen_t)

        # get text
        # get unique word indices
        # and the maximum number of oov words in the batch
        text = self.pointer_info.text[mask_t]
        word_indices = self.pointer_info.word_indices
        max_num_oov = self.pointer_info.max_num_oov
        new_vocab_size = (vocab_dist_t.size(0),vocab_dist_t.size(1)+max_num_oov)

        # create distrubtion over vocab using attention
        # NOTE: for the same words in the text, the attention probability is summed from those indices

        # indicator of size (# of unique words, batch size, seq length) such that
        # element (i, j, k) indicates whether unique word i is in batch example j at sequence position k
        indicator = text.expand(word_indices.size(0),*text.size()) == word_indices.view(-1,1,1).expand(-1,*text.size())

        # attention_indicator of size (# of unique words, batch size, seq length) such that
        # element (i, j, k) is the attention of batch example j at sequence position k if that word is unique word i else 0
        attention_indicator_t = torch.zeros(indicator.size(), device=indicator.device)
        attention_indicator_t[indicator] = attention_t.unsqueeze(0).expand(*indicator.size())[indicator]

        # sums up attention along each batch for each unique word
        # resulting in a matrix of size (batch size, # of unique words) where
        # element (i, j) expresses the probability mass in batch example i on unique word j
        # Note that the attention on a word after the sequence ends is 0 so we do not need to worry when we sum
        word_probabilities_t = torch.transpose(attention_indicator_t.sum(-1), 0, 1)

        add_at_indices = word_indices.expand(text.size(0),word_indices.size(0)).long()
        add_at_indices[add_at_indices < 0] += new_vocab_size[1].long()

        # attain mixture of the distributions according to p_gen

        # pad vocab distribution
        vocab_dist_probs_t = F.pad(vocab_dist_t, (0,max_num_oov))

        # get indices to add at
        final_vocab_dist_t = (p_gen_t*vocab_dist_probs_t).scatter_add(1, add_at_indices, (1-p_gen_t)*word_probabilities_t)

        return final_vocab_dist_t

    # this changes it so that only words that don't appear in the text and the static vocab are mapped to the oov index
    # used to get indices for computing loss
    # Note: DEPENDENT ON VALID INDICES IN POINTER_INFO
    def map_input_indices_(self, indices):
        # set oov not in text to oov index
        indices[indices < -self.pointer_info.max_num_oov] = len(self.vectorizer.word_vectors)
        oov_places = torch.nonzero(indices < 0)
        if oov_places.dim() > 1:
            batch_indices, oov_indices = oov_places[:,0], -1-indices[oov_places[:,0]]
            holes = self.pointer_info.oov_holes
            indices[batch_indices[holes[batch_indices, oov_indices.long()].byte()]] = len(self.vectorizer.word_vectors)

    def map_generated_indices_(self, indices):
        indices[indices >= len(self.vectorizer.word_vectors)] -= (len(self.vectorizer.word_vectors)+1+self.pointer_info.max_num_oov).numpy()

    # used to get indices to return after generating summary
    # Note: DEPENDENT ON CURRENT_P_GEN IN POINTER_INFO
    def get_extras(self):
        return (self.pointer_info.current_p_gen,)
