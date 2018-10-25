# Outline (I should probably split this file into shorter ones)
# 1. Sub modules for Summarization models
#         a) TextEncoder
#         b) ContextVectorNN
#         c) VocabularyDistributionNN
#         d) ProbabilityNN (only used in pointer-generator model)
# 2. Helper classes and functions for Summarization models
#         a) GeneratedSummary (used in generatring summaries during test time)
#         b) GeneratedSummaryHypothesis (used for beam search and subclasses Hypothesis and wraps GeneratedSummary object)
#         c) PointerInfo (used in pointer-generator model to keep track of extra info used by this model) (kind of hacky)
#         d) loss_function (this is trivial but is used by ModelManipulator in pytorch_helper.py)
#         e) error_function (this is also trivial but is used by ModelManipulator in pytorch_helper.py)
# 3. Summarization models
#         a) GeneratorModel
#         b) PointerGeneratorModel

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_helper import pack_padded_sequence_maintain_order, pad_packed_sequence_maintain_order, log_sum_exp
from utils import get_text_matrix
from beam_search import Hypothesis
import copy

################################ Sub modules for Summarization models ################################

# Encodes text through an LSTM
class TextEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TextEncoder, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        
    def forward(self, x, length):
        x, invert_indices = pack_padded_sequence_maintain_order(x, length, batch_first=True)
        output, (h, c) = self.lstm(x)
        output, (h, c) = pad_packed_sequence_maintain_order(output, [torch.transpose(h, 0, 1), torch.transpose(c, 0, 1)], invert_indices, batch_first=True)
        if output.sum() != output.sum():
            raise Exception
        return output, (h, c)
    
    @property
    def num_hidden(self):
        return self.lstm.hidden_size

# linear, activation, linear, softmax, sum where
# input is:
#     the hidden states from the TextEncoder, the current state from the SummaryDecoder
# and output is:
#     a softmax of a vector the length of the number of hidden states
class ContextVectorNN(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(ContextVectorNN, self).__init__()
        num_inputs = num_features+1
        self.conv1 = nn.Conv1d(num_inputs, num_hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(num_hidden, 1, kernel_size=1)
        
    def forward(self, text_states, text_length, summary_current_state, coverage):
        text_states = text_states.transpose(-1,-2)
        sizes = [-1]*text_states.dim()
        sizes[-1] = text_states.size(-1)
        summary_current_states = summary_current_state.unsqueeze(-1).expand(*sizes)
        outputs = torch.cat([text_states, summary_current_states, coverage], -2)
        outputs = torch.tanh(self.conv1(outputs))
        outputs = self.conv2(outputs)

        # indicator of elements that are within the length of that instance
        indicator = torch.arange(outputs.size(2), device=outputs.device).expand(*outputs.size()) < text_length.view(-1,1,1)
        unnormalized_log_attention = torch.zeros(outputs.size(), device=outputs.device) - float('inf')
        unnormalized_log_attention[indicator] = outputs[indicator]
        attention = F.softmax(unnormalized_log_attention, -1)
        
        context_vector = (attention*text_states).sum(-1)
        return context_vector, attention

# linear, log_softmax
# NOTE: paper says two linear lays with no activation between?
# NOTE: this returns a vocabulary distribution in log space, so
#       to get probabilities one most exponentiate
class VocabularyDistributionNN(nn.Module):
    def __init__(self, num_features, num_vocab):
        super(VocabularyDistributionNN, self).__init__()
        self.linear1 = nn.Linear(num_features, num_vocab)
        
    def forward(self, context_vector, summary_current_state):
        inputs = torch.cat((context_vector, summary_current_state), -1)
        outputs = F.log_softmax(self.linear1(inputs), -1)
        return outputs

# like the VocabularyDistributionNN, it takes as input the context vector and current state of the summary
# however, it produces a probability for each batch indicating how much to trust in the generated vocab dist
# versus the attention distribution over the words in the text
class ProbabilityNN(nn.Module):
    def __init__(self, num_features):
        super(ProbabilityNN, self).__init__()
        self.linear1 = nn.Linear(num_features, 1)
        
    def forward(self, context_vector, summary_current_state):
        inputs = torch.cat((context_vector, summary_current_state), -1)
        outputs = torch.sigmoid(self.linear1(inputs))
        return outputs
    
################################ Helper classes and functions for Summarization models ################################

class GeneratedSummary:
    def __init__(self, batch_length, device, start_index, end_index):
        self.summary = [torch.zeros((batch_length,1), device=device).long()+start_index]
        self.summary_length = torch.zeros(batch_length, device=device).long()-1
        self.valid_indices = torch.arange(batch_length, device=device)
        self.end_index = end_index
        self.loss_unnormalized = torch.zeros(batch_length, device=device)
        self.log_probs = []
        self.attentions = []
        
    def get_summary_t(self):
        return self.summary[-1], self.valid_indices
    
    def update(self, summary_tp1, loss_t, log_prob, attention):
        self.summary.append(summary_tp1.unsqueeze(-1))
        
        # add global log probability
        self.loss_unnormalized[self.valid_indices] += loss_t
        
        # get indices of instances that are not finished
        # and get indices of instances that are finished
        ending = (summary_tp1[self.valid_indices] == self.end_index)
        ended_indices = self.valid_indices[ending]
        self.valid_indices = self.valid_indices[ending == 0]
        
        # set summary length for ended time steps
        self.summary_length[ended_indices] = len(self.summary)
        
        self.log_probs.append(log_prob.cpu().detach().numpy())
        self.attentions.append(attention.cpu().detach().numpy())
        
    def loss(self):
        return self.loss_unnormalized.sum()/(len(self)-1)
        
    def is_done(self):
        return (self.summary_length >= 0).sum() == self.summary_length.size(0) or len(self.summary) > 300
        
    def return_info(self):
        return torch.cat(self.summary, -1).cpu().detach().numpy(), self.summary_length.cpu().detach().numpy(), self.log_probs, self.attentions
    
    def __len__(self):
        return len(self.summary)
    
class GeneratedSummaryHypothesis(Hypothesis):
    @staticmethod
    def get_top_k(hypotheses, k):
        # set attributes that stay the same
        model, text_states, text_length, beam_size = hypotheses[0].model, hypotheses[0].text_states, hypotheses[0].text_length, hypotheses[0].beam_size
        
        # create list of hypothesis-dependent attributes in a tuple
        hyp_attrs = pad_and_concat([()])
        raise NotImplementedError
    
    def __init__(self, model, generated_summary, text_states, text_length, h, c, coverage, beam_size):
        self.model = model
        self.generated_summary = generated_summary
        self.text_states = text_states
        self.text_length = text_length
        self.h, self.c = h, c
        self.coverage = coverage
        self.beam_size = beam_size
        
        self.batch_length = text_states.size(0)
        self.device = text_states.device
        
    def next_hypotheses(self):
        # set timestep words, valid indices
        summary_t, valid_indices = self.generated_summary.get_summary_t()

        # take a time step
        vocab_dist, h, c, attention, _ = self.model.timestep(valid_indices, summary_t, self.text_states, self.text_length, self.h, self.c, self.coverage)
        
        hypotheses = []
        word_indices = torch.topk(vocab_dist, self.beam_size, dim=1)[1]
        for i in range(self.beam_size):
            generated_summary_temp = copy.deepcopy(self.generated_summary)
            
            # generate next summary words
            summary_tp1 = torch.zeros(self.batch_length, device=self.device).long()
            summary_tp1[valid_indices] = word_indices[i]

            # calculate log prob, calculate covloss if aplicable, update coverage if aplicable
            log_prob = self.calculate_log_prob(vocab_dist, summary_tp1[valid_indices])
            if self.with_coverage:
                covloss = self.calculate_covloss(coverage[valid_indices], attention[valid_indices])
                coverage += attention
            
            # update global log prob
            loss_t = -log_prob + self.gamma*(covloss if self.with_coverage else 0)

            generated_summary_temp.update(summary_tp1, loss_t, log_prob, attention)
            
            # add this generated summary as another hypothesis
            hyp = GeneratedSummaryHypotheses(self.model, generated_summary_temp, self.text_states, self.text_length, self.h, self.c, self.coverage, self.loss_unnormalized, self.beam_size)
            hypotheses.append(hyp)
            
        return hypotheses
    
    def is_done(self):
        return self.generated_summary.is_done()
    
        
class PointerInfo:
    def __init__(self, text, text_oov_indices):
        self.text = text
        self.text_oov_indices = text_oov_indices
        self.oov_lengths = torch.tensor([len(oovs) for oovs in text_oov_indices], device=text.device).int()
        self.max_num_oov = torch.max(self.oov_lengths)
        self.word_indices = torch.unique(text)
        self.valid_indices = None
        
    def update_valid_indices(self, valid_indices):
        self.valid_indices = valid_indices
        
    def get_text(self):
        return self.text[self.valid_indices] if self.valid_indices is not None else text
    
    def get_oov_lengths(self):
        return self.oov_lengths[self.valid_indices] if self.valid_indices is not None else self.oov_lengths
    
def loss_function(loss):
    return loss

def error_function(loss):
    return None
    
################################ Summarization models ################################

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
        
    def forward(self, text, text_length, summary=None, summary_length=None, generate_algorithm='greedy'):
        # get batch with vectors from index batch
        text = [get_text_matrix(example[:text_length[i]], self.word_vectors, text.size(1))[0].unsqueeze(0) for i,example in enumerate(text)]
        text = torch.cat(text, 0)
        
        # run text through lstm encoder
        text_states, (h, c) = self.text_encoder(text, text_length)
        
        if summary is None:
            if generate_algorithm == 'greedy':
                return self.forward_generate_greedy(text_states, text_length, h[:,0], c[:,0])
            else:
                raise Exception
        else:
            return self.forward_supervised(text_states, text_length, h[:,0], c[:,0], summary, summary_length)
        
#     def forward_generate(self, text_states, text_length, h, c, beam_size=1):
#         # initialize
#         batch_length = text_states.size(0)
#         device = text_states.device
#         coverage = torch.zeros((batch_length, 1, text_states.size(1)), device=device)
#         loss_unnormalized = 0
#         h, c = h[:,0], c[:,0]
#         generated_summary = GeneratedSummary(batch_length, device, self.start_index, self.end_index)
#         hypothesis = GeneratedSummaryHypothesis(self, generated_summary, text_states, text_length, h, c, coverage, beam_size)

#         result = beam_search([hypothesis], beam_size)
#         generated_summary = result.hypothesis
        
#         return loss_unnormalized/(len(generated_summary)-1), generated_summary.return_info()

    # implements the forward pass of the decoder for generating summaries at test time
    # when beam search is finished this will become depricated and one will instead use beam search
    # with beam size of 1 for a greedy summary generation
    def forward_generate_greedy(self, text_states, text_length, h, c):
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        coverage = torch.zeros((batch_length, 1, text_states.size(1)), device=device)
        generated_summary = GeneratedSummary(batch_length, device, self.start_index, self.end_index)
        while not generated_summary.is_done():
            # set timestep words, valid indices
            summary_t, valid_indices = generated_summary.get_summary_t()
            
            # take a time step
            vocab_dist, h, c, attention, _ = self.timestep(valid_indices, summary_t, text_states, text_length, h, c, coverage)
            
            # generate next summary words
            summary_tp1 = torch.zeros(batch_length, device=device).long()
            summary_tp1[valid_indices] = torch.max(vocab_dist, 1)[1]
            
            # calculate log prob, calculate covloss if aplicable, update coverage if aplicable
            log_prob = self.calculate_log_prob(vocab_dist, summary_tp1[valid_indices])
            if self.with_coverage:
                covloss = self.calculate_covloss(coverage[valid_indices], attention[valid_indices])
                coverage += attention
            
            # update global log prob
            loss_t = -log_prob + self.gamma*(covloss if self.with_coverage else 0)
            
            generated_summary.update(summary_tp1, loss_t, log_prob, attention)
            
        return generated_summary.loss(), generated_summary.return_info()

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
            
        T = summary.size(1)-1
        return dict(loss=loss_unnormalized.sum()/T)
    
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
        return vocab_dist[torch.arange(summary_tp1.size(0)).long(),summary_tp1.long()]
    
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
    def forward(self, text, text_length, text_oov_indices, summary=None, summary_length=None, generate_algorithm='greedy'):
        self.pointer_info = PointerInfo(text, text_oov_indices)
        return_values = super(self.__class__, self).forward(text, text_length, summary=summary, summary_length=summary_length, generate_algorithm=generate_algorithm)
        self.pointer_info = None
        return return_values
    
    def forward_generate_greedy(self, *args, **kwargs):
        return_values = super(self.__class__, self).forward_generate_greedy(*args, **kwargs)
        summary = return_values[1][0]
        summary[summary >= len(self.word_vectors)] -= (len(self.word_vectors)+1+self.pointer_info.max_num_oov)
        return (*return_values, self.pointer_info.text_oov_indices)
    
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
        vocab_dist1, h_t, c_t, attention_t, context_vector = super(self.__class__, self).timestep_forward(summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t)
        new_vocab_size = (vocab_dist1.size(0),vocab_dist1.size(1)+max_num_oov)
        
        # pad vocab_dist1 with -infinities
        vocab_dist1_padded = torch.zeros(new_vocab_size, device=vocab_dist1.device)-float('inf')
        vocab_dist1_padded[:,:vocab_dist1.size(1)] = vocab_dist1
        
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
        
        # create this distribution by constructing a tensor of -infs of size (batch size, vocab size (including oov)) and then
        # for each batch and every word index in word_indices the corresponding log probability for that word in the batch
        # sets the vocab distribution at that word index
        # NOTE: -infs will not ever be used because if a word is truly oov, than it will go to the oov slot in the vocab dist
        #       and if the word is in the input text but oov for the static vocab, it will go to the corresponding index in the
        #       vocab distribution, which will be set
        vocab_dist2 = torch.zeros(new_vocab_size, device=vocab_dist1.device)-float('inf')
        vocab_dist2[torch.arange(text.size(0)).view(-1,1),
                    word_indices.expand(text.size(0),word_indices.size(0)).long()] = torch.log(word_probabilities)
        
        # get probability of generating vs copying
        p_gen = self.probability_layer(context_vector, h_t)
        
        # attain mixture of the distributions according to p_gen
        distributions = torch.cat((vocab_dist1_padded.view(1,*new_vocab_size), vocab_dist2.view(1,*new_vocab_size)), 0)
        weights = torch.cat(((1-p_gen).view(1,new_vocab_size[0],1), p_gen.view(1,new_vocab_size[0],1)), 0)
#         final_vocab_dist = log_sum_exp(distributions, 0, weights=weights)
        # Hacks off all of the oov terms that appeared in the text so there are no nans in the gradient
        final_vocab_dist = log_sum_exp(distributions[:, :, :vocab_dist1.size(1)], 0, weights=weights[:, :, :vocab_dist1.size(1)])
        
        return final_vocab_dist, h_t, c_t, attention_t, context_vector
    
    # this changes it so that only words that don't appear in the text and the static vocab are mapped to the oov index
#     def map_oov_indices(self, indices):
#         indices[(indices < -self.pointer_info.get_oov_lengths()).squeeze(0)] = len(self.word_vectors)
    