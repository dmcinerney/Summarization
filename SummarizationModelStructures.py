import torch
from torch import nn
from torch.nn import functional as F
from pytorch_helper import pack_padded_sequence_maintain_order, pad_packed_sequence_maintain_order

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
    def __init__(self, num_features, num_hidden, with_coverage=False):
        super(self.__class__, self).__init__()
        self.with_coverage = with_coverage
        num_inputs = num_features if not self.with_coverage else num_features+1
        self.conv1 = nn.Conv1d(num_inputs, num_hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(num_hidden, 1, kernel_size=1)
        
    def forward(self, text_states, summary_current_state, coverage=None):
        if self.with_coverage and coverage is None:
            raise Exception
        text_states = text_states.transpose(-1,-2)
        sizes = [-1]*text_states.dim()
        sizes[-1] = text_states.size(-1)
        summary_current_states = summary_current_state.unsqueeze(-1).expand(*sizes)
        input_vectors = [text_states, summary_current_states]
        if self.with_coverage:
            input_vectors.append(coverage)
        outputs = torch.cat(input_vectors, -2)
        outputs = torch.tanh(self.conv1(outputs))
        outputs = self.conv2(outputs)
        attention = F.softmax(outputs, -1)
        context_vector = (attention*text_states).sum(-1)
        return context_vector, attention

# linear, softmax
# NOTE: paper says two linear lays with no activation between?
class VocubularyDistributionNN(nn.Module):
    def __init__(self, num_features, num_vocab):
        super(self.__class__, self).__init__()
        self.linear1 = nn.Linear(num_features, num_vocab)
        
    def forward(self, context_vector, summary_current_state):
        outputs = torch.cat((context_vector, summary_current_state), -1)
        outputs = F.log_softmax(self.linear1(outputs), -1)
        return outputs