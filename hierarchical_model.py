import torch
from torch import nn
from torch.nn import functional as F
from beam_search import beam_search
from submodules import LSTMTextEncoder, CombineContext, LSTMSummaryDecoder, VocabularyDistributionNN, ProbabilityNN
from hierarchical_submodules import HierarchicalContextVectorNN, ReduceContextVector
from model_helpers import GeneratedSummary, PointerInfo, trim_text
from hierarchical_model_helpers import HierarchicalGeneratedSummaryHypothesis
import numpy as np
import parameters as p
import pdb

# Outline:
# a) Summarizer
# b) Encoder
# c) Decoder

def custom_trim_text(text, text_length, max_length):
    stacked_lengths = (text_length.float() @ torch.tensor(np.triu(np.ones((text_length.size(1), text_length.size(1)))), device=text.device).float()).int()
    mask = stacked_lengths < max_length
    after_mask = mask != torch.cat([torch.ones_like(mask[:,:1]), mask[:,:-1]], 1)
    length_change = max_length-stacked_lengths[after_mask]
    new_text_length = torch.zeros_like(text_length)
    new_text_length[mask] = text_length[mask]
    new_text_length[after_mask] = text_length[after_mask]+length_change.long()
    new_text = torch.zeros_like(text)
    text_mask = torch.arange(text.shape[2], device=text.device) < new_text_length.unsqueeze(2)
    new_text[text_mask] = text[text_mask]
    max_sentences = (new_text_length != 0).sum(1).max()
    max_sentence_length = new_text_length.max()
    return new_text[:,:max_sentences,:max_sentence_length], new_text_length[:,:max_sentences]

class Summarizer(nn.Module):
    def __init__(self, vectorizer, start_index, end_index, num_hidden=None, attn_hidden=None, with_coverage=False, gamma=1, with_pointer=False, sentence_encoder_base=LSTMTextEncoder, encoder_base=LSTMTextEncoder, decoder_base=LSTMSummaryDecoder, decoder_parallel_base=None):
        super(Summarizer, self).__init__()

        self.vectorizer = vectorizer
        self.start_index = start_index
        self.end_index = end_index
        self.num_hidden = vectorizer.vector_size//2 if num_hidden is None else num_hidden
        self.attn_hidden = attn_hidden
        self.with_coverage = with_coverage
        self.gamma = gamma
        self.with_pointer = with_pointer
        self.sentence_encoder_base = sentence_encoder_base
        self.encoder_base = encoder_base
        self.decoder_base = decoder_base
        self.decoder_parallel_base = decoder_parallel_base
        self.init_submodules()
        
        self.aspects = ['summary']

    def init_submodules(self):
        decoder_class = Decoder if not self.with_pointer else PointerGenDecoder
        self.encoder = Encoder(self.vectorizer, self.num_hidden, sentence_encoder_base=self.sentence_encoder_base, encoder_base=self.encoder_base)
        self.decoder = decoder_class(self.vectorizer, self.start_index, self.end_index, self.num_hidden, attn_hidden=self.attn_hidden, with_coverage=self.with_coverage, gamma=self.gamma, decoder_base=self.decoder_base, decoder_parallel_base=self.decoder_parallel_base)

    def forward(self, text, text_length, text_oov_indices=None, summary=None, summary_length=None, beam_size=1, store=None):
        text, text_length = custom_trim_text(text, text_length, p.MAX_TEXT_LENGTH)
        if summary is not None:
            summary, summary_length = trim_text(summary, summary_length, p.MAX_SUMMARY_LENGTH)
        text_states, sentence_states, state = self.encoder(text, text_length, store=store)
        if self.with_pointer:
            self.decoder.set_pointer_info(PointerInfo(text.contiguous().view(text.shape[0], text.shape[1]*text.shape[2]), text_oov_indices))
        return self.decoder(text_states, sentence_states, text_length, state, summary=summary, summary_length=summary_length, beam_size=beam_size)

class Encoder(nn.Module):
    def __init__(self, vectorizer, num_hidden, sentence_encoder_base=LSTMTextEncoder, encoder_base=LSTMTextEncoder):
        super(Encoder, self).__init__()
        self.vectorizer = vectorizer
        num_features = self.vectorizer.vector_size
        self.num_hidden = num_hidden*2
        self.sentence_encoder = sentence_encoder_base(num_features, self.num_hidden)
        self.text_encoder = encoder_base(self.num_hidden, self.num_hidden)

    def forward(self, text, text_length, store=None):
        # get batch with vectors from index batch
#         text = [self.vectorizer.get_text_matrix(example[:text_length[i]], text.size(1))[0].unsqueeze(0) for i,example in enumerate(text)]
#         text = torch.cat(text, 0)
        batch_size, num_sentences, num_words = text.shape
        flattened_text = text.contiguous().view(-1, num_words)
        flattened_text_length = text_length.contiguous().view(-1)
        word_vectors = self.vectorizer(flattened_text, flattened_text_length)

        # run text through lstm encoder
        non_zero_mask = flattened_text_length.view(-1) != 0
        word_states = torch.zeros((batch_size*num_sentences, num_words, self.num_hidden), device=text.device)
        state = torch.zeros((batch_size*num_sentences, self.num_hidden), device=text.device)
        word_states[non_zero_mask], state[non_zero_mask] = self.sentence_encoder(word_vectors[non_zero_mask], flattened_text_length[non_zero_mask], store=store)
        word_states = word_states.view(batch_size, num_sentences, num_words, self.num_hidden)
        sentence_states = state.view(batch_size, num_sentences, self.num_hidden)
        num_sentences = (text_length != 0).sum(1)
        sentence_states, state = self.text_encoder(sentence_states, num_sentences, store=store)
        return word_states, sentence_states, state

class Decoder(nn.Module):
    def __init__(self, vectorizer, start_index, end_index, num_hidden, attn_hidden=None, with_coverage=False, gamma=1, decoder_base=LSTMSummaryDecoder, decoder_parallel_base=None):
        super(Decoder, self).__init__()
        self.vectorizer = vectorizer
        self.start_index = start_index
        self.end_index = end_index
        self.num_hidden = num_hidden
        self.attn_hidden = num_hidden//2 if attn_hidden is None else attn_hidden
        self.with_coverage = with_coverage
        self.gamma = gamma

        self.num_features = self.vectorizer.vector_size
        self.num_vocab = self.vectorizer.vocab_size
        self.decoder_base = decoder_base
        self.decoder_parallel_base = decoder_parallel_base
        self.init_submodules()

    def init_submodules(self):
        self.combine_context = CombineContext(self.num_features, self.num_hidden*2)
        self.summary_decoder = self.decoder_base(self.num_features, self.num_hidden*2)
        if self.decoder_parallel_base is not None:
#             self.summary_decoder_parallel = self.decoder_parallel_base(self.summary_decoder)
            raise NotImplementedError("Parallel base optimized mode is still under construction!")
        self.context_nn = HierarchicalContextVectorNN(self.num_hidden*4+1, self.attn_hidden)
        self.reduce_context = ReduceContextVector(self.num_hidden*4)
        self.vocab_nn = VocabularyDistributionNN(self.num_hidden*4, self.num_hidden, self.num_vocab+1)

    def forward(self, text_states, sentence_states, text_length, state, summary=None, summary_length=None, beam_size=1):
        if summary is None:
            return self.decode_generate(text_states, sentence_states, text_length, state, beam_size=beam_size)
        else:
            return self.decode_train(text_states, sentence_states, text_length, state, summary, summary_length)
#             return self.decode_train_optimized(text_states, text_length, state, summary, summary_length)

    def decode_generate(self, text_states, sentence_states, text_length, state, beam_size=1):
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        generated_summary = GeneratedSummary(batch_length, device, self.start_index, self.end_index)
        coverage = torch.zeros((batch_length, text_states.size(1)*text_states.size(2)), device=device)
        context_vector = torch.zeros((batch_length, text_states.size(3)), device=device)
        hypothesis = HierarchicalGeneratedSummaryHypothesis(self, generated_summary, text_states, sentence_states, text_length, state, coverage, context_vector)

        summary_hyps = beam_search(hypothesis.next_hypotheses(beam_size), beam_size)
        results = [summary_hyp.generated_summary.return_info() for summary_hyp in summary_hyps]

        for r in results:
            indices = r[0]
            self.map_generated_indices_(indices)

        return [results]

    # implements the forward pass of the decoder for training
    # this uses teacher forcing, but conceivably one could try
    # and should add other algorithms for training
    def decode_train(self, text_states, sentence_states, text_length, state, summary, summary_length):
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        coverage = torch.zeros((batch_length, text_states.size(1)*text_states.size(2)), device=device)
        context_vector = torch.zeros((batch_length, text_states.size(3)), device=device)
        loss_unnormalized = torch.zeros(batch_length, device=device)
        summary_tp1 = summary[:,0]
        for t in range(summary.size(1)-1):
            # set timestep words
            summary_t = summary_tp1

            # get indices of instances that are not finished
            valid_indices = torch.nonzero((summary_length-t-1) > 0)[:,0]

            # take a time step
            vocab_dist, state, attention, context_vector = self.timestep_wrapper(valid_indices, summary_t, text_states, sentence_states, text_length, state, coverage, context_vector)

            # get next time step words
            summary_tp1 = summary[:,t+1]

            # calculate log prob, calculate covloss if aplicable, update coverage if aplicable
            summary_tp1_valid = summary_tp1[valid_indices]
            self.map_input_indices_(summary_tp1_valid)
            log_prob = torch.zeros(batch_length, device=device)
            log_prob[valid_indices] = self.calculate_log_prob(vocab_dist, summary_tp1_valid)
            if self.with_coverage:
                covloss = torch.zeros(batch_length, device=device)
                covloss[valid_indices] = self.calculate_covloss(coverage[valid_indices], attention[valid_indices])
                coverage += attention

            # update unnormalized loss
            loss_unnormalized += -log_prob + self.gamma*(covloss if self.with_coverage else 0)

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

    # this timestep calls timestep forward and converts the inputs to and from just the valid batch examples
    # of those inputs at that time step
    def timestep_wrapper(self, valid_indices, summary_t, text_states, sentence_states, text_length, prev_state, coverage, prev_context_vector):
        # create tensors for returned values that need to have first dim of size batch_size
        attention = torch.zeros_like(coverage, device=coverage.device)
        context_vector = torch.zeros_like(prev_context_vector, device=prev_context_vector.device)
        # NOTE: vocab_dist is returned with a first dim size of valid_indices.size(0)
        #   because we never need it to be of the full batch size

        # do forward pass
        vocab_dist, _, state_temp, attention[valid_indices], context_vector[valid_indices] = self.timestep(summary_t[valid_indices], text_states[valid_indices], sentence_states[valid_indices], text_length[valid_indices], prev_state[valid_indices], coverage[valid_indices], prev_context_vector[valid_indices])

        # create new state of full batch size (need to do this afterwards because it could be variable length
        #   so need to get new sizes from state_temp)
        state = torch.zeros((prev_state.size(0), *state_temp.shape[1:]), device=state_temp.device)
        state[valid_indices] = state_temp
        # vocab_dist = vocab_dist + p.EPSILON
        # vocab_dist = vocab_dist/vocab_dist.sum(1, keepdim=True)
        return vocab_dist, state, attention, context_vector

    # runs the inputs for a time step through the neural nets to get the vocab distribution for that timestep
    # and other necessary information: inputs to the next hidden state in the decoder, attention, and the context vector
    # (the context vector is only needed in the subclass of this so kinda bad style but whatever)
    def timestep(self, summary_t, text_states, sentence_states, text_length, prev_state, coverage, prev_context_vector):
        # summary_vec_t = self.vectorizer.get_text_matrix(summary_t, len(summary_t))[0]
        summary_vec_t = self.vectorizer(summary_t.view(1,-1), torch.tensor([len(summary_t)], device=summary_t.device))[0]
        
        summary_vec_t_mod = self.combine_context(summary_vec_t, prev_context_vector)
        output, state = self.summary_decoder(summary_vec_t_mod, prev_state)
        context_vector, attention = self.context_nn(text_states, sentence_states, text_length, output.unsqueeze(1), coverage)
        context_vector, attention = context_vector[:,0], attention[:,0]
        context_vector = self.reduce_context(context_vector)
        vocab_dist = self.vocab_nn(context_vector, output)
        return vocab_dist, output, state, attention, context_vector

    # calulates the log probability of the summary at a time step given the vocab distribution for that time step
    def calculate_log_prob(self, vocab_dist, summary_tp1):
        return torch.log(vocab_dist[torch.arange(summary_tp1.size(0)).long(),summary_tp1.long()])

    # calculates the coverage loss for each batch example at a time step
    def calculate_covloss(self, coverage, attention):
        return torch.min(torch.cat((coverage.unsqueeze(0), attention.unsqueeze(0)), 0), 0)[0].sum(1)

    # map oov indices maps the indices of oov words to a specific index corresponding to the position in the vocab distribution
    # that represents an oov word
    def map_input_indices_(self, indices):
        indices[indices == -1] = self.vectorizer.vocab_size

    def map_generated_indices_(self, indices):
        indices[indices == self.vectorizer.vocab_size] = -1

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
        self.probability_layer = ProbabilityNN(self.num_hidden*4)

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


    def timestep_wrapper(self, valid_indices, summary_t, text_states, sentence_states, text_length, prev_state, coverage, prev_context_vector):
        self.pointer_info.update_valid_indices(valid_indices)
        return super(PointerGenDecoder, self).timestep_wrapper(valid_indices, summary_t, text_states, sentence_states, text_length, prev_state, coverage, prev_context_vector)

    def timestep(self, summary_t, text_states, sentence_states, text_length, prev_state, coverage, prev_context_vector):
        if self.pointer_info is None:
            raise Exception

        # execute the normal timestep_forward function
        vocab_dist, output, state, attention, context_vector = super(PointerGenDecoder, self).timestep(summary_t, text_states, sentence_states, text_length, prev_state, coverage, prev_context_vector)

        final_vocab_dist = self.timestep_addon(vocab_dist, output, attention, context_vector)

        return final_vocab_dist, output, state, attention, context_vector

    def timestep_addon(self, vocab_dist, output, attention, context_vector):
        # get probability of generating vs copying
        if p.P_GEN is None:
            p_gen = self.probability_layer(context_vector, output)
        else:
            p_gen = torch.zeros((context_vector.size(0),1), device=context_vector.device) + p.P_GEN
        self.pointer_info.update_p_gen(p_gen)

        # get text
        # get unique word indices
        # and the maximum number of oov words in the batch
        text = self.pointer_info.get_text()
        word_indices = self.pointer_info.word_indices
        max_num_oov = self.pointer_info.max_num_oov
        new_vocab_size = (vocab_dist.size(0),vocab_dist.size(1)+max_num_oov)
        # create distrubtion over vocab using attention
        # NOTE: for the same words in the text, the attention probability is summed from those indices

        # indicator of size (# of unique words, batch size, seq length) such that
        # element (i, j, k) indicates whether unique word i is in batch example j at sequence position k
        indicator = text.expand(word_indices.size(0),*text.size()) == word_indices.view(-1,1,1).expand(-1,*text.size())

        # attention_indicator of size (# of unique words, batch size, seq length) such that
        # element (i, j, k) is the attention of batch example j at sequence position k if that word is unique word i else 0
        #attention_indicator = torch.zeros(indicator.size(), device=indicator.device)
        #attention_indicator[indicator] = attention.unsqueeze(0).expand(*indicator.size())[indicator]

        # sums up attention along each batch for each unique word
        # resulting in a matrix of size (batch size, # of unique words) where
        # element (i, j) expresses the probability mass in batch example i on unique word j
        # Note that the attention on a word after the sequence ends is 0 so we do not need to worry when we sum
        #word_probabilities = torch.transpose(attention_indicator.sum(-1), 0, 1)
        
        word_probabilities = torch.einsum('ijk,jk->ji', indicator.float(), attention)

        add_at_indices = word_indices.expand(text.size(0),word_indices.size(0)).long()
        add_at_indices[add_at_indices < 0] += new_vocab_size[1].long()

        # attain mixture of the distributions according to p_gen

        # pad vocab distribution
        vocab_dist_probs = F.pad(vocab_dist, (0,max_num_oov))

        # get indices to add at
        final_vocab_dist = (p_gen*vocab_dist_probs).scatter_add(1, add_at_indices, (1-p_gen)*word_probabilities)

        return final_vocab_dist

    # this changes it so that only words that don't appear in the text and the static vocab are mapped to the oov index
    # used to get indices for computing loss
    # Note: DEPENDENT ON VALID INDICES IN POINTER_INFO
    # THIS IS SUPER HACKY
    def map_input_indices_(self, indices):
        # set oov not in text to oov index
        indices[indices < -self.pointer_info.max_num_oov] = self.vectorizer.vocab_size
        oov_places = torch.nonzero(indices < 0)
        if oov_places.dim() > 1:
            batch_indices, oov_indices = oov_places[:,0], -1-indices[oov_places[:,0]]
            holes = self.pointer_info.get_oov_holes()
            if holes is not None:
                indices[batch_indices[holes[batch_indices, oov_indices.long()].byte()]] = self.vectorizer.vocab_size

    def map_generated_indices_(self, indices):
        indices[indices >= self.vectorizer.vocab_size] -= (self.vectorizer.vocab_size+1+self.pointer_info.max_num_oov).cpu().numpy()

    # used to get indices to return after generating summary
    # Note: DEPENDENT ON CURRENT_P_GEN IN POINTER_INFO
    def get_extras(self):
        return (self.pointer_info.current_p_gen,)
