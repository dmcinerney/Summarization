import torch
from torch import nn
from torch.nn import functional as F
from utils import get_text_matrix
from beam_search import beam_search
from submodules import TextEncoder, ContextVectorNN, VocabularyDistributionNN, ProbabilityNN
from model_helpers import GeneratedSummary, GeneratedSummaryHypothesis, PointerInfo

# Outline:
# a) GeneratorModel
# b) PointerGeneratorModel

class GeneratorModel(nn.Module):
    def __init__(self, word_vectors, start_index, end_index, num_hidden1=None, num_hidden2=None, with_coverage=False, gamma=1):
        super(GeneratorModel, self).__init__()
        self.word_vectors = word_vectors
        num_features = len(self.word_vectors[0])
        num_vocab = len(self.word_vectors)
        self.start_index = start_index
        self.end_index = end_index
        self.num_hidden1 = num_features//2 if num_hidden1 is None else num_hidden1
        self.num_hidden2 = num_features//2 if num_hidden2 is None else num_hidden2
        self.with_coverage = with_coverage
        self.gamma = gamma
        
        self.text_encoder = TextEncoder(num_features, self.num_hidden1, bidirectional=True)
        self.summary_decoder = nn.LSTMCell(num_features, self.num_hidden1)
        self.context_nn = ContextVectorNN(self.num_hidden1*3, self.num_hidden2)
        self.vocab_nn = VocabularyDistributionNN(self.num_hidden1*3, num_vocab+1)
        
    def forward(self, text, text_length, summary=None, summary_length=None, beam_size=1):
        # get batch with vectors from index batch
        text = [get_text_matrix(example[:text_length[i]], self.word_vectors, text.size(1))[0].unsqueeze(0) for i,example in enumerate(text)]
        text = torch.cat(text, 0)
        
        # run text through lstm encoder
        text_states, (h, c) = self.text_encoder(text, text_length)
        
        if summary is None:
            return self.forward_generate(text_states, text_length, h[:,0], c[:,0], beam_size=beam_size)
        else:
            return self.forward_supervised(text_states, text_length, h[:,0], c[:,0], summary, summary_length)
        
    def forward_generate(self, text_states, text_length, h, c, beam_size=1):
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        generated_summary = GeneratedSummary(batch_length, device, self.start_index, self.end_index)
        coverage = torch.zeros((batch_length, 1, text_states.size(1)), device=device)
        hypothesis = GeneratedSummaryHypothesis(self, generated_summary, text_states, text_length, h, c, coverage)
        
        results = beam_search(hypothesis.next_hypotheses(beam_size), beam_size)
        
        return [(r.generated_summary.loss(), r.generated_summary.return_info()) for r in results]

    # implements the forward pass of the decoder for training
    # this uses teacher forcing, but conceivably one could try
    # and should add other algorithms for training
    def forward_supervised(self, text_states, text_length, h, c, summary, summary_length):
        if summary_length is None:
            raise Exception
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        coverage = torch.zeros((batch_length, 1, text_states.size(1)), device=device)
        loss_unnormalized = torch.zeros(batch_length, device=device)
        summary_tp1 = summary[:,0]
        for t in range(summary.size(1)-1):
            # set timestep words
            summary_t = summary_tp1
            
            # get indices of instances that are not finished
            valid_indices = torch.nonzero((summary_length-t-1) > 0)[:,0]
            
            # take a time step
            vocab_dist, h, c, attention, _ = self.timestep(valid_indices, summary_t, text_states, text_length, h, c, coverage)
            
            # get next time step words
            summary_tp1 = summary[:,t+1]
            
            # calculate log prob, calculate covloss if aplicable, update coverage if aplicable
            log_prob = self.calculate_log_prob(vocab_dist, summary_tp1[valid_indices])
            if self.with_coverage:
                covloss = self.calculate_covloss(coverage[valid_indices], attention[valid_indices])
                coverage += attention
            
            # update unnormalized loss
            loss_unnormalized[valid_indices] += -log_prob + self.gamma*(covloss if self.with_coverage else 0)
            
        return dict(loss=(loss_unnormalized/(summary_length.float()-1)).sum())
    
    # this timestep calls timestep forward and converts the inputs to and from just the valid batch examples
    # of those inputs at that time step
    def timestep(self, valid_indices, summary_t, text_states, text_length, h, c, coverage):
        # inputs at valid indices at position t
        text_states_t = text_states[valid_indices]
        text_length_t = text_length[valid_indices]
        summary_t = summary_t[valid_indices]
        h_t, c_t = h[valid_indices], c[valid_indices]
        coverage_t = coverage[valid_indices]
        
        # do forward pass
        vocab_dist, h_t, c_t, attention_t, context_vector = self.timestep_forward(summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t)
        
        # set new h, c, coverage, and loss
        h[valid_indices], c[valid_indices] = h_t, c_t
        attention = torch.zeros_like(coverage, device=h.device)
        attention[valid_indices] = attention_t
        return vocab_dist, h, c, attention, context_vector
    
    # runs the inputs for a time step through the neural nets to get the vocab distribution for that timestep
    # and other necessary information: inputs to the next hidden state in the decoder, attention, and the context vector
    # (the context vector is only needed in the subclass of this so kinda bad style but whatever)
    def timestep_forward(self, summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t):
        summary_vec_t = get_text_matrix(summary_t, self.word_vectors, len(summary_t))[0]
        
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
        return torch.min(torch.cat((coverage, attention), -2), -2)[0].sum(-1)
    
    # map oov indices maps the indices of oov words to a specific index corresponding to the position in the vocab distribution
    # that represents an oov word
    def map_oov_indices(self, indices):
        indices[indices < 0] = -1
    
    
# This model subclasses the generator model so that on each forward timestep, it averages the generator vocab distribution
# with a pointer distribution obtained from the attention distribution and these are weighted by p_gen and 1-p_gen respectively
# where p_gen is the probability of generating vs copying
class PointerGeneratorModel(GeneratorModel):
    def __init__(self, word_vectors, start_index, end_index, num_hidden1=None, num_hidden2=None, with_coverage=False, gamma=1):
        super(PointerGeneratorModel, self).__init__(word_vectors, start_index, end_index, num_hidden1=num_hidden1, num_hidden2=num_hidden2, with_coverage=with_coverage, gamma=gamma)
        self.probability_layer = ProbabilityNN(self.num_hidden1*3)
        self.pointer_info = None
    
    # this is a little bit of a hacky solution, setting the pointer info as an object attribute temporarily
    def forward(self, text, text_length, text_oov_indices, summary=None, summary_length=None, beam_size=1):
        self.pointer_info = PointerInfo(text, text_oov_indices)
        return_values = super(self.__class__, self).forward(text, text_length, summary=summary, summary_length=summary_length, beam_size=beam_size)
        self.pointer_info = None
        return return_values
    
    def forward_generate(self, *args, **kwargs):
        return_values = super(self.__class__, self).forward_generate(*args, **kwargs)
        for v in return_values:
            summary = v[1][0]
            summary[summary >= len(self.word_vectors)] -= (len(self.word_vectors)+1+self.pointer_info.max_num_oov)
        return return_values, self.pointer_info.text_oov_indices
    
    def timestep(self, valid_indices, summary_t, text_states, text_length, h, c, coverage):
        self.pointer_info.update_valid_indices(valid_indices)
        return_values = super(self.__class__, self).timestep(valid_indices, summary_t, text_states, text_length, h, c, coverage)
        return return_values
    
    def timestep_forward(self, summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t):
        if self.pointer_info is None:
            raise Exception
            
        # get text
        # get unique word indices
        # and the maximum number of oov words in the batch
        text = self.pointer_info.get_text()
        word_indices = self.pointer_info.word_indices
        max_num_oov = self.pointer_info.max_num_oov
        
        # execute the normal timestep_forward function
        vocab_dist, h_t, c_t, attention_t, context_vector = super(self.__class__, self).timestep_forward(summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t)
        new_vocab_size = (vocab_dist.size(0),vocab_dist.size(1)+max_num_oov)
        
        # create distrubtion over vocab using attention
        # NOTE: for the same words in the text, the attention probability is summed from those indices
        
        # indicator of size (# of unique words, batch size, seq length) such that
        # element (i, j, k) indicates whether unique word i is in batch example j at sequence position k
        indicator = text.expand(word_indices.size(0),*text.size()) == word_indices.view(-1,1,1).expand(-1,*text.size())
        
        # attention_indicator of size (# of unique words, batch size, seq length) such that
        # element (i, j, k) is the attention of batch example j at sequence position k if that word is unique word i else 0
        attention_indicator = torch.zeros(indicator.size(), device=indicator.device)
        attention_indicator[indicator] = torch.transpose(attention_t, 0, 1).expand(*indicator.size())[indicator]
        
        # sums up attention along each batch for each unique word
        # resulting in a matrix of size (batch size, # of unique words) where
        # element (i, j) expresses the probability mass in batch example i on unique word j
        # Note that the attention on a word after the sequence ends is 0 so we do not need to worry when we sum
        word_probabilities = torch.transpose(attention_indicator.sum(-1), 0, 1)
        
        # get probability of generating vs copying
        p_gen = self.probability_layer(context_vector, h_t)
        
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
        indices[(indices.int() < -self.pointer_info.get_oov_lengths()).squeeze(0)] = len(self.word_vectors)
    