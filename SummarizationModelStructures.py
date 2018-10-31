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
from pytorch_helper import pack_padded_sequence_maintain_order, pad_packed_sequence_maintain_order, log_sum_exp, pad_and_concat, batch_stitch
from utils import get_text_matrix
from beam_search import Hypothesis, beam_search
import copy

import pdb

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
        inputs = torch.cat([text_states, summary_current_states, coverage], -2)
        scores = self.conv2(torch.tanh(self.conv1(inputs)))

        # indicator of elements that are within the length of that instance
        indicator = torch.arange(scores.size(2), device=scores.device).expand(*scores.size()) >= text_length.view(-1,1,1)
        unnormalized_log_attention = scores.masked_scatter(indicator, torch.zeros(indicator.sum(), device=scores.device)-float('inf'))
        attention = F.softmax(unnormalized_log_attention, -1)
#         # indicator of elements that are within the length of that instance
#         indicator = torch.arange(scores.size(2), device=scores.device).expand(*scores.size()) < text_length.view(-1,1,1)
#         attention = F.softmax(scores, -1)*indicator.float()
#         attention = attention/attention.sum(2, keepdim=True)
        
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
        outputs = F.softmax(self.linear1(inputs), -1)
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
    @classmethod
    def batch_stitch(cls, generated_summaries, indices):
        # concatenate all of the relevant attributes into a list (all elements should be tensors)
        summary_list = [gs.summary for gs in generated_summaries]
        summary_length_list = [gs.summary_length for gs in generated_summaries]
        loss_unnormalized_list = [gs.loss_unnormalized for gs in generated_summaries]
        log_probs_list = [gs.log_probs for gs in generated_summaries]
        attentions_list = [gs.attentions for gs in generated_summaries]
        
        # use batch_stitch to get the resultant attribute tensors of size (indices.size(0), batch_length, etc...)
        summary_list, summary_length_list, loss_unnormalized_list, log_probs_list, attentions_list = batch_stitch(
            [summary_list,
             summary_length_list,
             loss_unnormalized_list,
             log_probs_list,
             attentions_list],
            indices,
            static_flags=[False,True,True,False,False]
        )
        
        curr_length = max(len(gs) for gs in generated_summaries)
        new_generated_summaries = []
        for i in range(indices.size(0)):
            # reduce padding to only what is needed (if summary is not done, then this is just the current length)
            max_length = torch.max(summary_length_list[i]) if not (summary_length_list[i] == -1).any() else curr_length
            # create new summary and set the state to the state of the stitched batch
            new_generated_summary = GeneratedSummary(
                end_index=generated_summaries[0].end_index,
                summary=summary_list[i, :, :max_length],
                summary_length=summary_length_list[i],
                loss_unnormalized=loss_unnormalized_list[i],
                log_probs=log_probs_list[i, :, :(max_length-1)],
                attentions=attentions_list[i, :, :(max_length-1)]
            )
            
            new_generated_summaries.append(new_generated_summary)
            
        return new_generated_summaries
    
    def __init__(self, batch_length=None, device=None, start_index=None, end_index=None, summary=None, summary_length=None, valid_indices=None, loss_unnormalized=None, log_probs=None, attentions=None):
        self.summary = torch.zeros((batch_length,1), device=device).long()+start_index if summary is None else summary
        self.summary_length = (torch.zeros(batch_length, device=device).long()-1) if summary_length is None else summary_length
        self.valid_indices = torch.arange(self.summary_length.size(0), device=self.summary_length.device)[self.summary_length < 0]
        self.end_index = end_index
        self.loss_unnormalized = torch.zeros(batch_length, device=device) if loss_unnormalized is None else loss_unnormalized
        self.log_probs = torch.zeros(0, device=device) if log_probs is None else log_probs
        self.attentions = torch.zeros(0, device=device) if attentions is None else attentions
        
    def get_summary_t(self):
        return self.summary[:,-1], self.valid_indices
    
    def update(self, summary_tp1, loss_t, log_prob, attention):
        self.summary = torch.cat((self.summary, summary_tp1.unsqueeze(-1)), -1)
        
        # add loss for each batch
        self.loss_unnormalized[self.valid_indices] += loss_t
        
        # expand log prob to full batch dimention
        full_log_prob = torch.zeros(self.summary_length.size(0), device=self.summary_length.device).scatter(0, self.valid_indices, log_prob).unsqueeze(-1)
        # append log_prob to log_probs
        self.log_probs = torch.cat((self.log_probs, full_log_prob), -1)
        # append attention to attentions
        self.attentions = torch.cat((self.attentions, attention.unsqueeze(-1)), -1)
        
        # get indices of instances that are not finished
        # and get indices of instances that are finished
        ending = (summary_tp1[self.valid_indices] == self.end_index)
        ended_indices = self.valid_indices[ending]
        self.valid_indices = self.valid_indices[ending == 0]
        
        # set summary length for ended time steps
        self.summary_length[ended_indices] = self.summary.size(1)
        
    def loss(self):
        return self.loss_unnormalized/(self.length().float()-1)
        
    def is_done(self):
        return (self.summary_length >= 0).sum() == self.summary_length.size(0) or len(self) > 300
        
    def return_info(self):
        return self.summary.cpu().detach().numpy(), self.summary_length.cpu().detach().numpy(), self.log_probs.cpu().detach().numpy(), self.attentions.cpu().detach().numpy()
    
    def length(self):
        length = torch.tensor(self.summary_length)
        length[length < 0] = len(self)
        return length
    
    def __len__(self):
        return self.summary.size(1)
    
    def copy(self):
        gs_copy = GeneratedSummary(
            end_index=self.end_index,
            summary=torch.tensor(self.summary),
            summary_length=torch.tensor(self.summary_length),
            loss_unnormalized=torch.tensor(self.loss_unnormalized),
            log_probs=torch.tensor(self.log_probs),
            attentions=torch.tensor(self.attentions)
        )
        return gs_copy
    
class GeneratedSummaryHypothesis(Hypothesis):
    @classmethod
    def get_top_k(cls, hypotheses, k, sorted=False):
        losses = pad_and_concat([hyp.generated_summary.loss() for hyp in hypotheses], static=True)
        indices = torch.topk(losses, k, dim=0, largest=False, sorted=sorted)[1]
        return cls.batch_stitch(hypotheses, indices)
    
    @classmethod
    def batch_stitch(cls, hypotheses, indices):
        # set attributes that stay the same
        model, text_states, text_length = hypotheses[0].model, hypotheses[0].text_states, hypotheses[0].text_length
        
        # call the stitch function on all non-tensor attributes that differ
        generated_summaries = GeneratedSummary.batch_stitch([hyp.generated_summary for hyp in hypotheses], indices)
        
        # create tensors of all of the tensor attributes that differ
        h_list = [hyp.h for hyp in hypotheses]
        c_list = [hyp.c for hyp in hypotheses]
        coverage_list = [hyp.coverage for hyp in hypotheses]
        h_list, c_list, coverage_list = batch_stitch(
            [h_list, c_list, coverage_list],
            indices,
            static_flags=[True, True, True]
        )
        
        return [cls(model, generated_summaries[i], text_states, text_length, h_list[i], c_list[i], coverage_list[i]) for i in range(h_list.size(0))]
        
    def __init__(self, model, generated_summary, text_states, text_length, h, c, coverage):
        self.model = model
        self.generated_summary = generated_summary
        self.text_states = text_states
        self.text_length = text_length
        self.h, self.c = h, c
        self.coverage = coverage
        
        self.batch_length = text_states.size(0)
        self.device = text_states.device
        
    def next_hypotheses(self, beam_size):
        # set timestep words, valid indices
        summary_t, valid_indices = self.generated_summary.get_summary_t()

        # take a time step
        vocab_dist, h, c, attention, _ = self.model.timestep(valid_indices, summary_t, self.text_states, self.text_length, self.h, self.c, self.coverage)
        
        hypotheses = []
        word_indices = torch.topk(vocab_dist, beam_size, dim=1)[1]
        for i in range(beam_size):
            generated_summary_temp = self.generated_summary.copy()
            
            # generate next summary words
            summary_tp1 = torch.zeros(self.batch_length, device=self.device).long()
            summary_tp1[valid_indices] = word_indices[:,i]

            # calculate log prob, calculate covloss if aplicable, update coverage if aplicable
            log_prob = self.model.calculate_log_prob(vocab_dist, summary_tp1[valid_indices])
            if self.model.with_coverage:
                covloss = self.model.calculate_covloss(coverage[valid_indices], attention[valid_indices])
                coverage += attention
            
            # update global log prob
            loss_t = -log_prob + self.model.gamma*(covloss if self.model.with_coverage else 0)

            generated_summary_temp.update(summary_tp1, loss_t, log_prob, attention)
            
            # add this generated summary as another hypothesis
            hyp = GeneratedSummaryHypothesis(self.model, generated_summary_temp, self.text_states, self.text_length, self.h, self.c, self.coverage)
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
    return loss.sum()

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
        
    def forward(self, text, text_length, summary=None, summary_length=None, beam_size=1, return_all=False):
        # get batch with vectors from index batch
        text = [get_text_matrix(example[:text_length[i]], self.word_vectors, text.size(1))[0].unsqueeze(0) for i,example in enumerate(text)]
        text = torch.cat(text, 0)
        
        # run text through lstm encoder
        text_states, (h, c) = self.text_encoder(text, text_length)
        
        if summary is None:
            return self.forward_generate(text_states, text_length, h[:,0], c[:,0], beam_size=beam_size, return_all=return_all)
#             return self.forward_generate_greedy(text_states, text_length, h[:,0], c[:,0])
        else:
            return self.forward_supervised(text_states, text_length, h[:,0], c[:,0], summary, summary_length)
        
    def forward_generate(self, text_states, text_length, h, c, beam_size=1, return_all=False):
        # initialize
        batch_length = text_states.size(0)
        device = text_states.device
        generated_summary = GeneratedSummary(batch_length, device, self.start_index, self.end_index)
        coverage = torch.zeros((batch_length, 1, text_states.size(1)), device=device)
        hypothesis = GeneratedSummaryHypothesis(self, generated_summary, text_states, text_length, h, c, coverage)
        
        results = beam_search(hypothesis.next_hypotheses(beam_size), beam_size)
        
        if not return_all:
            generated_summary = results[0].generated_summary
            return generated_summary.loss(), generated_summary.return_info()
        else:
            return [(r.generated_summary.loss(), r.generated_summary.return_info()) for r in results]

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
    def forward(self, text, text_length, text_oov_indices, summary=None, summary_length=None, beam_size=1, return_all=False):
        self.pointer_info = PointerInfo(text, text_oov_indices)
        return_values = super(self.__class__, self).forward(text, text_length, summary=summary, summary_length=summary_length, beam_size=1, return_all=False)
        self.pointer_info = None
        return return_values
    
    def forward_generate(self, *args, **kwargs):
        return_values = super(self.__class__, self).forward_generate(*args, **kwargs)
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
        indices[(indices < -self.pointer_info.get_oov_lengths()).squeeze(0)] = len(self.word_vectors)
    