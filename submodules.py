import torch
from torch import nn
from torch.nn import functional as F
from pytorch_helper import pack_padded_sequence_maintain_order, pad_packed_sequence_maintain_order

# Description: this file contains the sub neural networks that are used in the summarization models
# Outline:
# a) TextEncoder
# b) ContextVectorNN
# c) VocabularyDistributionNN
# d) ProbabilityNN (only used in pointer-generator model)

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
# and outputs are:
#     context_vector (a weighted sum of the encoder hidden states according to the attention)
#     attention (a softmax of a vector the length of the number of hidden states)
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
        indicator = torch.arange(scores.size(2), device=scores.device).expand(*scores.size()) < text_length.view(-1,1,1)
        attention = F.softmax(scores, -1)*indicator.float()
        attention = attention/attention.sum(2, keepdim=True)
        
        context_vector = (attention*text_states).sum(-1)
        return context_vector, attention

# linear, softmax
# NOTE: paper says two linear lays to reduce parameters!
class VocabularyDistributionNN(nn.Module):
    def __init__(self, num_features, num_vocab):
        super(VocabularyDistributionNN, self).__init__()
        self.linear1 = nn.Linear(num_features, num_vocab)
        
    def forward(self, context_vector, summary_current_state):
        inputs = torch.cat((context_vector, summary_current_state), -1)
        outputs = F.softmax(self.linear1(inputs), -1)
        return outputs

# like the VocabularyDistributionNN, it takes as input the context vector and current state of the summary
# however, it produces a probability for each batch indicating how much weight to put on the generated vocab
# dist versus the attention distribution over the words in the text when creating the final vocab distribution
class ProbabilityNN(nn.Module):
    def __init__(self, num_features):
        super(ProbabilityNN, self).__init__()
        self.linear1 = nn.Linear(num_features, 1)
        
    def forward(self, context_vector, summary_current_state):
        inputs = torch.cat((context_vector, summary_current_state), -1)
        outputs = torch.sigmoid(self.linear1(inputs))
        return outputs