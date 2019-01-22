import torch
from torch import nn
from torch.nn import functional as F
from pytorch_helper import pack_padded_sequence_maintain_order, pad_packed_sequence_maintain_order
import parameters as p
from model_helpers import init_lstm_weights
from transformer import positional_encoding, CustomTransformer, CustomTransformerCell
from attention import Attention, AdditiveAttention

# Description: this file contains the sub neural networks that are used in the summarization models
# Outline:
# a) TextEncoder
# b) ContextVectorNN
# c) VocabularyDistributionNN
# d) ProbabilityNN (only used in pointer-generator model)

# Encodes text through an LSTM
class LSTMTextEncoder(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(LSTMTextEncoder, self).__init__()
        self.lstm = nn.LSTM(num_features, num_hidden, bidirectional=True, batch_first=True)
        init_lstm_weights(self.lstm)
        self.state_encoder = StateEncoder(num_hidden)

    def forward(self, x, length):
        x, invert_indices = pack_padded_sequence_maintain_order(x, length, batch_first=True)
        output, (h, c) = self.lstm(x)
        output, (h, c) = pad_packed_sequence_maintain_order(output, [torch.transpose(h, 0, 1), torch.transpose(c, 0, 1)], invert_indices, batch_first=True)
        return output, torch.cat(self.state_encoder(h, c), 1)

# This class will be used to encode the h and c output of the text
# in order to be input into the decoder
class StateEncoder(nn.Module):
    def __init__(self, num_hidden):
        super(StateEncoder, self).__init__()
        self.linearh = nn.Linear(num_hidden*2, num_hidden)
        self.linearc = nn.Linear(num_hidden*2, num_hidden)
#         for param in self.parameters():
#             param.data.normal_(std=p.WEIGHT_INIT_STD)

    def forward(self, h, c):
        h1, h2 = h[:, 0], h[:, 1]
        c1, c2 = c[:, 0], c[:, 1]
        h = F.relu(self.linearh(torch.cat((h1, h2), 1)))
        c = F.relu(self.linearc(torch.cat((c1, c2), 1)))
        return h, c

class TransformerTextEncoder(nn.Module):
    def __init__(self, num_features, num_hidden, N=6):
        super(TransformerTextEncoder, self).__init__()
        num_heads = p.NUM_TRANSFORMER_HEADS
        num_hidden *= 2
        self.linear = nn.Linear(num_features, num_hidden)
        self.transformers = nn.ModuleList([CustomTransformer(num_hidden, num_heads) for _ in range(N)])

    def forward(self, x, length):
        x = x + positional_encoding(sequence_length=x.size(1), vector_length=x.size(2), device=x.device)
        x = self.linear(x)
        for transformer in self.transformers:
            x, distribution = transformer(x, length)
        return x, None

class LSTMSummaryDecoder(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(LSTMSummaryDecoder, self).__init__()
        self.lstm_cell = nn.LSTMCell(num_features, num_hidden)

    def forward(self, token, previous):
        half = previous.size(1)//2
        h, c = previous[:,:half], previous[:,half:]
        h, c = self.lstm_cell(token, (h, c))
        return torch.cat([h, c], 1), torch.cat([h, c], 1)

class TransformerSummaryDecoder(nn.Module):
    def __init__(self, num_features, num_hidden, N=6):
        super(TransformerSummaryDecoder, self).__init__()
        num_heads = p.NUM_TRANSFORMER_HEADS
        num_hidden *= 2
        self.linear = nn.Linear(num_features, num_hidden)
        self.transformer_cells = nn.ModuleList([CustomTransformerCell(num_hidden, num_heads) for _ in range(N)])

    # token - batch_size, vector_length
    # previous - batch_size, num_layers, sequence_length, vector_length
    def forward(self, token, previous):
        length = previous.size(2) if previous is not None else 0
        # TODO: slightly inefficient
        token = token + positional_encoding(sequence_length=length+1, vector_length=token.size(1), device=token.device)[:,-1,:]
        token = self.linear(token)
        next_previous_list = []
        for i,transformer_cell in enumerate(self.transformer_cells):
            token, next_previous, distribution = transformer_cell(token, previous[:,i] if previous is not None else None)
            next_previous_list.append(next_previous.unsqueeze(1))
        return token, torch.cat(next_previous_list, 1)

# linear, activation, linear, softmax, sum where
# input is:
#     the hidden states from the TextEncoder, the current state from the SummaryDecoder
# and outputs are:
#     context_vector (a weighted sum of the encoder hidden states according to the attention)
#     attention (a softmax of a vector the length of the number of hidden states)
class ContextVectorNN(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(ContextVectorNN, self).__init__()
#         self.linear1 = nn.Linear(num_inputs, num_hidden)
#         self.linear2 = nn.Linear(num_hidden, 1)
        self.additive_attention = AdditiveAttention(num_inputs, num_hidden)

    def forward(self, text_states, text_length, summary_current_state, coverage):
        coverages = coverage.unsqueeze(2) # batch_size, sequence_length, 1

#         # OLD ATTENTION
#         summary_current_states = summary_current_state.unsqueeze(1).expand(*text_states.shape[:2],summary_current_state.size(1))
#             # batch_size, sequence_length, q_vector_length
#         inputs = torch.cat((text_states, summary_current_states, coverages), 2)
#             # batch_size, sequence_length, ts_vector_length+q_vector_length+1

#         scores = self.linear2(torch.tanh(self.linear1(inputs))).squeeze(2)

#         # indicator of elements that are within the length of that instance
#         indicator = torch.arange(scores.size(1), device=scores.device).view(1,-1) < text_length.view(-1,1)
#         attention = F.softmax(scores, 1)*indicator.float()
#         attention = attention/attention.sum(1, keepdim=True)

#         context_vector = (attention.unsqueeze(2)*text_states).sum(1)

        # NEW ATTENTION
        keys = torch.cat([text_states, coverages], 2) # batch_size, sequence_length, ts_vector_length+1
        mask = torch.arange(keys.size(1), device=keys.device).unsqueeze(0) < text_length.unsqueeze(1)
            # batch_size, sequence_length
        queries = summary_current_state
        context_vector, attention = self.additive_attention(queries, keys, text_states, mask=mask.unsqueeze(1), return_distribution=True)

        return context_vector, attention

# linear, softmax
# NOTE: paper says two linear lays to reduce parameters!
class VocabularyDistributionNN(nn.Module):
    def __init__(self, num_features, num_hidden, num_vocab):
        super(VocabularyDistributionNN, self).__init__()
        self.linear1 = nn.Linear(num_features, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_vocab)

    def forward(self, context_vector, summary_current_state):
        inputs = torch.cat((context_vector, summary_current_state), -1)
        outputs = F.softmax(self.linear2(self.linear1(inputs)), -1)
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
