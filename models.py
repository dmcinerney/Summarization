import torch
from torch import nn
from torch.nn import functional as F
from beam_search import beam_search
from submodules import TextEncoder, StateEncoder, ContextVectorNN, VocabularyDistributionNN, ProbabilityNN
from model_helpers import GeneratedSummary, GeneratedSummaryHypothesis, PointerInfo, init_lstm_weights
import pdb

# Outline:
# a) Summarizer
# b) Encoder
# c) Decoder

class Summarizer(nn.Module):
    def __init__(self, vectorizer, start_index, end_index, lstm_hidden=None, attn_hidden=None, with_coverage=False, gamma=1, with_pointer=False):
        super(Summarizer, self).__init__()

        self.with_pointer = with_pointer
        lstm_hidden = len(vectorizer.word_vectors[0])//2 if lstm_hidden is None else lstm_hidden
        decoder_class = Decoder if not self.with_pointer else PointerGenDecoder

        self.encoder = Encoder(vectorizer, lstm_hidden)
        self.decoder = decoder_class(vectorizer, start_index, end_index, lstm_hidden, attn_hidden=attn_hidden, with_coverage=with_coverage, gamma=gamma)

    def forward(self, text, text_length, text_oov_indices=None, summary=None, summary_length=None, beam_size=1):
        text_states, (h, c) = self.encoder(text, text_length)
        if self.with_pointer:
            self.decoder.set_pointer_info(PointerInfo(text, text_oov_indices))
        return self.decoder(text_states, text_length, h, c, summary=summary, summary_length=summary_length, beam_size=beam_size)

class Encoder(nn.Module):
    def __init__(self, vectorizer, lstm_hidden):
        super(Encoder, self).__init__()
        self.vectorizer = vectorizer
        num_features = len(self.vectorizer.word_vectors[0])

        self.text_encoder = TextEncoder(num_features, lstm_hidden)
        self.state_encoder = StateEncoder(lstm_hidden)

    def forward(self, text, text_length):
        # get batch with vectors from index batch
        text = [self.vectorizer.get_text_matrix(example[:text_length[i]], text.size(1))[0].unsqueeze(0) for i,example in enumerate(text)]
        text = torch.cat(text, 0)

        # run text through lstm encoder
        text_states, (h, c) = self.text_encoder(text, text_length)
        h, c = self.state_encoder(h, c)

        return text_states, (h, c)

class Decoder(nn.Module):
    def __init__(self, vectorizer, start_index, end_index, lstm_hidden, attn_hidden=None, with_coverage=False, gamma=1):
        super(Decoder, self).__init__()
        self.vectorizer = vectorizer
        num_features = len(self.vectorizer.word_vectors[0])
        num_vocab = len(self.vectorizer.word_vectors)
        self.start_index = start_index
        self.end_index = end_index
        attn_hidden = num_features//2 if attn_hidden is None else attn_hidden
        self.with_coverage = with_coverage
        self.gamma = gamma

        self.summary_decoder = nn.LSTMCell(num_features, lstm_hidden)
        init_lstm_weights(self.summary_decoder)
        self.context_nn = ContextVectorNN(lstm_hidden*3+1, attn_hidden)
        self.vocab_nn = VocabularyDistributionNN(lstm_hidden*3, num_vocab+1)

    def forward(self, text_states, text_length, h, c, summary=None, summary_length=None, beam_size=1):
        if summary is None:
            return self.decode_generate(text_states, text_length, h, c, beam_size=beam_size)
        else:
            return self.decode_train(text_states, text_length, h, c, summary, summary_length)

    def decode_generate(self, text_states, text_length, h, c, beam_size=1):
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        generated_summary = GeneratedSummary(batch_length, device, self.start_index, self.end_index)
        coverage = torch.zeros((batch_length, text_states.size(1)), device=device)
        hypothesis = GeneratedSummaryHypothesis(self, generated_summary, text_states, text_length, h, c, coverage)

        results = beam_search(hypothesis.next_hypotheses(beam_size), beam_size)

        return [r.generated_summary.return_info() for r in results]

    # implements the forward pass of the decoder for training
    # this uses teacher forcing, but conceivably one could try
    # and should add other algorithms for training
    def decode_train(self, text_states, text_length, h, c, summary, summary_length):
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        coverage = torch.zeros((batch_length, text_states.size(1)), device=device)
        loss_unnormalized = torch.zeros(batch_length, device=device)
        summary_tp1 = summary[:,0]
        for t in range(summary.size(1)-1):
            # set timestep words
            summary_t = summary_tp1

            # get indices of instances that are not finished
            valid_indices = torch.nonzero((summary_length-t-1) > 0)[:,0]

            # take a time step
            vocab_dist, h, c, attention, _ = self.timestep_wrapper(valid_indices, summary_t, text_states, text_length, h, c, coverage)

            # get next time step words
            summary_tp1 = summary[:,t+1]

            # calculate log prob, calculate covloss if aplicable, update coverage if aplicable
            log_prob = torch.zeros(batch_length, device=device)
            log_prob[valid_indices] = self.calculate_log_prob(vocab_dist, summary_tp1[valid_indices])
            if self.with_coverage:
                covloss = torch.zeros(batch_length, device=device)
                covloss[valid_indices] = self.calculate_covloss(coverage[valid_indices], attention[valid_indices])
                coverage += attention

            # update unnormalized loss
            loss_unnormalized += -log_prob + self.gamma*(covloss if self.with_coverage else 0)

        return dict(loss=(loss_unnormalized/(summary_length.float()-1)))

    # this timestep calls timestep forward and converts the inputs to and from just the valid batch examples
    # of those inputs at that time step
    def timestep_wrapper(self, valid_indices, summary_t, text_states, text_length, h, c, coverage):
        # inputs at valid indices at position t
        text_states_t = text_states[valid_indices]
        text_length_t = text_length[valid_indices]
        summary_t = summary_t[valid_indices]
        h_t, c_t = h[valid_indices], c[valid_indices]
        coverage_t = coverage[valid_indices]

        # do forward pass
        vocab_dist, h_t, c_t, attention_t, context_vector = self.timestep(summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t)

        # set new h, c, coverage, and loss
        h[valid_indices], c[valid_indices] = h_t, c_t
        attention = torch.zeros_like(coverage, device=h.device)
        attention[valid_indices] = attention_t
        return vocab_dist, h, c, attention, context_vector

    # runs the inputs for a time step through the neural nets to get the vocab distribution for that timestep
    # and other necessary information: inputs to the next hidden state in the decoder, attention, and the context vector
    # (the context vector is only needed in the subclass of this so kinda bad style but whatever)
    def timestep(self, summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t):
        summary_vec_t = self.vectorizer.get_text_matrix(summary_t, len(summary_t))[0]

        h_t, c_t = self.summary_decoder(summary_vec_t, (h_t, c_t))
        context_vector, attention_t = self.context_nn(text_states_t, text_length_t, h_t, coverage_t)
        vocab_dist = self.vocab_nn(context_vector, h_t)
        return vocab_dist, h_t, c_t, attention_t, context_vector

    # calulates the log probability of the summary at a time step given the vocab distribution for that time step
    def calculate_log_prob(self, vocab_dist, summary_tp1):
        self.map_oov_indices(summary_tp1)
        return torch.log(vocab_dist[torch.arange(summary_tp1.size(0)).long(),summary_tp1.long()])

    # calculates the coverage loss for each batch example at a time step
    def calculate_covloss(self, coverage, attention):
        return torch.min(torch.cat((coverage.unsqueeze(0), attention.unsqueeze(0)), 0), 0)[0].sum(1)

    # map oov indices maps the indices of oov words to a specific index corresponding to the position in the vocab distribution
    # that represents an oov word
    def map_oov_indices(self, indices):
        indices[indices < 0] = -1

    # this adds any extra information you may want to add to a summary
    def get_extras(self):
        return ()

# This model subclasses the generator model so that on each forward timestep, it averages the generator vocab distribution
# with a pointer distribution obtained from the attention distribution and these are weighted by p_gen and 1-p_gen respectively
# where p_gen is the probability of generating vs copying
class PointerGenDecoder(Decoder):
    def __init__(self, *args, **kwargs):
        super(PointerGenDecoder, self).__init__(*args, **kwargs)
        lstm_hidden = args[3]
        self.probability_layer = ProbabilityNN(lstm_hidden*3)
        self.pointer_info = None

    def set_pointer_info(self, pointer_info):
        self.pointer_info = pointer_info

    # this is a little bit of a hacky solution, setting the pointer info as an object attribute temporarily
    def forward(self, *args, **kwargs):
        return_values = super(PointerGenDecoder, self).forward(*args, **kwargs)
        self.pointer_info = None
        return return_values

    def decode_generate(self, *args, **kwargs):
        return_values = super(PointerGenDecoder, self).decode_generate(*args, **kwargs)
        for v in return_values:
            summary = v[0]
            summary[summary >= len(self.vectorizer.word_vectors)] -= (len(self.vectorizer.word_vectors)+1+self.pointer_info.max_num_oov)
        return return_values

    def timestep_wrapper(self, valid_indices, summary_t, text_states, text_length, h, c, coverage):
        self.pointer_info.update_valid_indices(valid_indices)
        return_values = super(PointerGenDecoder, self).timestep_wrapper(valid_indices, summary_t, text_states, text_length, h, c, coverage)
        return return_values

    def timestep(self, summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t):
        if self.pointer_info is None:
            raise Exception

        # get text
        # get unique word indices
        # and the maximum number of oov words in the batch
        text = self.pointer_info.get_text()
        word_indices = self.pointer_info.word_indices
        max_num_oov = self.pointer_info.max_num_oov

        # execute the normal timestep_forward function
        vocab_dist, h_t, c_t, attention_t, context_vector = super(PointerGenDecoder, self).timestep(summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t)
        new_vocab_size = (vocab_dist.size(0),vocab_dist.size(1)+max_num_oov)

        # create distrubtion over vocab using attention
        # NOTE: for the same words in the text, the attention probability is summed from those indices

        # indicator of size (# of unique words, batch size, seq length) such that
        # element (i, j, k) indicates whether unique word i is in batch example j at sequence position k
        indicator = text.expand(word_indices.size(0),*text.size()) == word_indices.view(-1,1,1).expand(-1,*text.size())

        # attention_indicator of size (# of unique words, batch size, seq length) such that
        # element (i, j, k) is the attention of batch example j at sequence position k if that word is unique word i else 0
        attention_indicator = torch.zeros(indicator.size(), device=indicator.device)
        attention_indicator[indicator] = attention_t.unsqueeze(0).expand(*indicator.size())[indicator]

        # sums up attention along each batch for each unique word
        # resulting in a matrix of size (batch size, # of unique words) where
        # element (i, j) expresses the probability mass in batch example i on unique word j
        # Note that the attention on a word after the sequence ends is 0 so we do not need to worry when we sum
        word_probabilities = torch.transpose(attention_indicator.sum(-1), 0, 1)

        # get probability of generating vs copying
        p_gen = self.probability_layer(context_vector, h_t)
        self.pointer_info.update_p_gen(p_gen)

        # attain mixture of the distributions according to p_gen

        # pad vocab distribution
        vocab_dist_probs = F.pad(vocab_dist, (0,max_num_oov))

        # get indices to add at
        add_at_indices = word_indices.expand(text.size(0),word_indices.size(0)).long()
        add_at_indices[add_at_indices < 0] += new_vocab_size[1].long()
        final_vocab_dist = (p_gen*vocab_dist_probs).scatter_add(1, add_at_indices, (1-p_gen)*word_probabilities)

        return final_vocab_dist, h_t, c_t, attention_t, context_vector

    # this changes it so that only words that don't appear in the text and the static vocab are mapped to the oov index
    def map_oov_indices(self, indices):
        indices[(indices.int() < -self.pointer_info.get_oov_lengths()).squeeze(0)] = len(self.vectorizer.word_vectors)

    def get_extras(self):
        return (self.pointer_info.current_p_gen,)
