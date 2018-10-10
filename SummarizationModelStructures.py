import torch
from torch import nn
from torch.nn import functional as F
from pytorch_helper import pack_padded_sequence_maintain_order, pad_packed_sequence_maintain_order
from utils import get_text_matrix

class TextEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs)
        
    def forward(self, x, length):
        x, invert_indices = pack_padded_sequence_maintain_order(x, length, batch_first=True)
        output, (h, c) = self.lstm(x)
        output, (h, c) = pad_packed_sequence_maintain_order(output, [torch.transpose(h, 0, 1), torch.transpose(c, 0, 1)], invert_indices, batch_first=True)
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
        super(self.__class__, self).__init__()
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
        
        # softmax each batch individually because they have different lengths
        max_length = outputs.size(2)
        attention = torch.cat([F.pad(F.softmax(batch[:,:text_length[i]].unsqueeze(0),-1), (0,max_length-text_length[i])) for i,batch in enumerate(outputs)], 0)
        
        context_vector = (attention*text_states).sum(-1)
        return context_vector, attention

# linear, softmax
# NOTE: paper says two linear lays with no activation between?
class VocubularyDistributionNN(nn.Module):
    def __init__(self, num_features, num_vocab):
        super(self.__class__, self).__init__()
        self.linear1 = nn.Linear(num_features, num_vocab)
        
    def forward(self, context_vector, summary_current_state):
        inputs = torch.cat((context_vector, summary_current_state), -1)
        outputs = F.log_softmax(self.linear1(inputs), -1)
        return outputs
    
class ProbabilityNN(nn.Module):
    def __init__(self, num_features):
        super(self.__class__, self).__init__()
        self.linear1 = nn.Linear(num_features, 1)
        
    def forward(self, context_vector, summary_current_state):
        inputs = torch.cat((context_vector, summary_current_state), -1)
        outputs = torch.sigmoid(self.linear1(inputs))
        return outputs
    
class GeneratorModel(nn.Module):
    def __init__(self, word_vectors, start_index, end_index, num_hidden1=None, num_hidden2=None, with_coverage=False, gamma=1):
        super(self.__class__, self).__init__()
        self.word_vectors = word_vectors
        num_features = len(self.word_vectors[0])
        num_vocab = len(self.word_vectors)
        self.start_index = start_index
        self.end_index = end_index
        if num_hidden1 is None:
            num_hidden1 = num_features//2
        if num_hidden2 is None:
            num_hidden2 = num_features//2
        self.with_coverage = with_coverage
        self.gamma = gamma
        
        self.text_encoder = TextEncoder(num_features, num_hidden1, bidirectional=True)
        self.summary_decoder = nn.LSTMCell(num_features, num_hidden1)
        self.context_nn = ContextVectorNN(num_hidden1*3, num_hidden2)
        self.vocab_nn = VocubularyDistributionNN(num_hidden1*3, num_vocab)
        
    def forward(self, text, text_length, summary=None, summary_length=None, generate_algorithm='greedy'):
        # get batch with vectors from index batch
        text = [get_text_matrix(example[:text_length[i]], self.word_vectors, text.size(1))[0].unsqueeze(0) for i,example in enumerate(text)]
        text = torch.cat(text, 0)
        
        # run text through lstm encoder
        text_states, (h, c) = self.text_encoder(text, text_length)
        
        if summary is None:
            if generate_algorithm == 'greedy':
                return self.forward_generate_greedy(text_states, text_length, h, c)
            else:
                raise Exception
        else:
            return self.forward_supervised(text_states, text_length, h, c, summary, summary_length)
    
    def forward_generate_greedy(self, text_states, text_length, h, c):
        # initialize
        coverage = torch.zeros((text_states.size(0), 1, text_states.size(1)), device=text_states.device)
        loss_unnormalized = 0
        h, c = h[:,0], c[:,0]
        batch_length = h.size(0)
        summary = [torch.zeros((batch_length,1), device=h.device).long()+self.start_index]
        summary_length = torch.zeros(batch_length, device=h.device).long()-1
        valid_indices = torch.arange(batch_length, device=h.device)
        t = 0
        while True:
            # set timestep words
            summary_t = summary[-1]
            
            # take a time step
            vocab_dist, h, c, attention = self.timestep(valid_indices, summary_t, text_states, text_length, h, c, coverage)
            
            # get next time step words
            summary_tp1 = torch.zeros(batch_length, device=h.device).long()
            summary_tp1[valid_indices] = torch.max(vocab_dist, 1)[1]
            summary.append(summary_tp1.unsqueeze(-1))
            
            # calculate loss, coverage
            loss_unnormalized += self.calculate_loss(vocab_dist, summary_tp1[valid_indices], coverage[valid_indices], attention[valid_indices])
            if self.with_coverage:
                coverage += attention
            
            # get indices of instances that are not finished
            # and get indices of instances that are finished
            ending = (summary_tp1[valid_indices] == self.end_index)
            ended_indices = valid_indices[torch.nonzero(ending).squeeze(-1)]
            valid_indices = valid_indices[torch.nonzero(ending == 0).squeeze(-1)]
            
            t += 1
            
            # set summary length for ended time steps
            summary_length[ended_indices] = t+1
            
            # check if all summaries have ended
            if (summary_length >= 0).sum() == summary_length.size(0) or t > 300:
                break
        
        return loss_unnormalized/t, torch.cat(summary, 1), summary_length
    
    # not done yet
    def forward_generate_beam(self, text_states, text_length, h, c, beam_width=1):
        # initialize
        coverage = torch.zeros((text_states.size(0), 1, text_states.size(1)), device=text_states.device)
        loss_unnormalized = RunningAverage()
        h, c = h[:,0], c[:,0]
        batch_length = h.size(0)
        summary = [torch.zeros((batch_length,1), device=h.device).long()+self.start_index]
        summary_length = torch.zeros(batch_length, device=h.device).long()-1
        states = [(summary, summary_length, h, c, coverage, loss)]
        valid_indices = torch.arange(batch_length, device=h.device)
        t = 0
        while True:
            # set timestep words
            summary_t = summary[-1]
            
            # take a time step
            vocab_dist, h, c, attention = self.timestep(valid_indices, summary_t, text_states, text_length, h, c, coverage)
            
            # get next time step words
            summary_tp1 = torch.zeros(batch_length, device=h.device).long()
            summary_tp1[valid_indices] = torch.max(vocab_dist, 1)[1]
            summary.append(summary_tp1.unsqueeze(-1))
            
            # calculate loss, coverage
            loss_unnormalized += self.calculate_loss(vocab_dist, summary_tp1[valid_indices], coverage[valid_indices], attention[valid_indices])
            if self.with_coverage:
                coverage += attention
            
            # get indices of instances that are not finished
            # and get indices of instances that are finished
            ending = (summary_tp1[valid_indices] == self.end_index)
            ended_indices = valid_indices[torch.nonzero(ending).squeeze(-1)]
            valid_indices = valid_indices[torch.nonzero(ending == 0).squeeze(-1)]
            
            t += 1
            
            # set summary length for ended time steps
            summary_length[ended_indices] = t+1
            
            # check if all summaries have ended
            if (summary_length >= 0).sum() == summary_length.size(0) or t > 300:
                break
        
        return loss_unnormalized/t, torch.cat(summary, 1), summary_length


    def forward_supervised(self, text_states, text_length, h, c, summary, summary_length):
        if summary_length is None:
            raise Exception
        # initialize
        coverage = torch.zeros((text_states.size(0), 1, text_states.size(1)), device=text_states.device)
        loss_unnormalized = 0
        h, c = h[:,0], c[:,0]
        summary_tp1 = summary[:,0]
        for t in range(summary.size(1)-1):
            # set timestep words
            summary_t = summary_tp1
            
            # get indices of instances that are not finished
            valid_indices = torch.nonzero((summary_length-t-1) > 0)[:,0]
            
            # take a time step
            vocab_dist, h, c, attention = self.timestep(valid_indices, summary_t, text_states, text_length, h, c, coverage)
            
            # get next time step words
            summary_tp1 = summary[:,t+1]
            
            # calculate loss, coverage
            loss_unnormalized += self.calculate_loss(vocab_dist, summary_tp1[valid_indices], coverage[valid_indices], attention[valid_indices])
            if coverage is not None:
                coverage += attention
            
        T = summary.size(1)-1
        return dict(loss=loss_unnormalized/T)
        
    def timestep(self, valid_indices, summary_t, text_states, text_length, h, c, coverage):
        # inputs at valid indices at position t
        text_states_t = text_states[valid_indices]
        text_length_t = text_length[valid_indices]
        summary_t = summary_t[valid_indices]
        h_t, c_t = h[valid_indices], c[valid_indices]
        coverage_t = coverage[valid_indices]
        
        # do forward pass
        vocab_dist, h_t, c_t, attention_t = self.timestep_forward(summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t)
        
        # set new h, c, coverage, and loss
        h[valid_indices], c[valid_indices] = h_t, c_t
        attention = torch.zeros_like(coverage, device=h.device)
        attention[valid_indices] = attention_t
        return vocab_dist, h, c, attention
    
    def timestep_forward(self, summary_t, text_states_t, text_length_t, h_t, c_t, coverage_t):
        summary_vec_t = get_text_matrix(summary_t, self.word_vectors, len(summary_t))[0]
        
        h_t, c_t = self.summary_decoder(summary_vec_t, (h_t, c_t))
        context_vector, attention_t = self.context_nn(text_states_t, text_length_t, h_t, coverage_t)
        vocab_dist = self.vocab_nn(context_vector, h_t)
        return vocab_dist, h_t, c_t, attention_t

    def calculate_loss(self, vocab_dist, summary_tp1, coverage, attention):
        coverage_loss = torch.min(torch.cat((coverage, attention), -2), -2)[0].sum() if self.with_coverage else 0
        return -vocab_dist[torch.arange(summary_tp1.size(0)).long(),summary_tp1.long()].sum() + self.gamma*coverage_loss
    
    
    
class PointerGeneratorModel(GeneratorModel):
    def __init__(self, word_vectors, start_index, end_index, num_hidden1=None, num_hidden2=None, with_coverage=False, gamma=1):
        super(self.__class__, self).__init__(word_vectors, start_index, end_index, num_hidden1=None, num_hidden2=None, with_coverage=False, gamma=1)
        self.probability_layer = ProbabilityNN(num_hidden1*3)
        self.pointer_info = None
    
    # this is a little bit of a hacky solution, setting the oov indices as an object attribute temporarily
    def forward(self, text, text_length, text_oov_indices, summary=None, summary_length=None, generate_algorithm='greedy'):
        self.pointer_info = (text, text_oov_indices)
        return_values = super(self.__class__, self).forward(text, text_length, summary=None, summary_length=None, generate_algorithm='greedy')
        self.pointer_info = None
        return return_values
    
    def forward_generate_greedy(self, *args, **kwargs):
        loss, summary, summary_length = super(self.__class__, self).forward_generate_greedy(*args, **kwargs)
        return loss, summary, summary_length, self.pointer_info[1]
    
    def timestep_forward(self, summary_t, text_states_t, h_t, c_t, coverage_t):
        if text_oov_indices is None:
            raise Exception
        # get the maximum number of oov words for the instances in the batch
        max_num_oov = -torch.min(self.pointer_info[0])
        
        # execute the normal timestep_forward function
        # (need to rewrite the function because we need the context vector)
        summary_vec_t = get_text_matrix(summary_t, self.word_vectors, len(summary_t))[0]
        
        h_t, c_t = self.summary_decoder(summary_vec_t, (h_t, c_t))
        context_vector, attention_t = self.context_nn(text_states_t, h_t, coverage_t)
        vocab_dist1 = self.vocab_nn(context_vector, h_t)

        # pad this vocab distribution to the size of the final vocab dist
        vocab_dist1 = F.pad(vocab_dist1, (0,max_num_oov))
        
        # 
        vocab_dist2 = torch.zeros((vocab_dist1.size(0),vocab_dist1.size(1)+max_num_oov), device=vocab_dist1.device)
        batch_indices = torch.arange(vocab_dist.size(0)).view(1,-1)
        vocab_dist[batch_indices, text] += 
        
        p = self.probability_layer(context_vector, h_t)
        
#         return vocab_dist, h_t, c_t, attention_t