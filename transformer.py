import torch
from torch import nn
import math
from attention import ScaledDotProductAttention

# taken from http://nlp.seas.harvard.edu/2018/04/03/attention.html
def positional_encoding(sequence_length, vector_length, device):
    pe = torch.zeros(sequence_length, vector_length, device=device)
    position = torch.arange(0, sequence_length, device=device).unsqueeze(1).float()
    div_term1 = torch.exp(torch.arange(0, vector_length, 2, device=device).float() *
                         -(math.log(10000.0) / vector_length))
    div_term2 = torch.exp((torch.arange(1, vector_length, 2, device=device).float()-1) *
                         -(math.log(10000.0) / vector_length))
    pe[:, 0::2] = torch.sin(position * div_term1)
    pe[:, 1::2] = torch.cos(position * div_term2)
    return pe.unsqueeze(0)

class Transformer(nn.Module):
    def __init__(self, attention_func, unidirectional=False):
        super(Transformer, self).__init__()
        self.attention_func = attention_func
        self.unidirectional = unidirectional

    def forward(self, x, length):
        arangement = torch.arange(x.size(1), device=x.device)
        mask = arangement.unsqueeze(0) < length.unsqueeze(1) # batch_size, sequence_length
        mask = mask.unsqueeze(1) # batch_size, 1, sequence_length
        if self.unidirectional:
            unimask = arangement.unsqueeze(0) <= arangement.unsqueeze(1) # sequence_length, sequence_length
            mask = mask*unimask.unsqueeze(0) # batch_size, sequence_length, sequence_length
        return self.attention_func(x, x, x, mask=mask) # batch_size, sequence_length, vector_length

class TransformerCell(nn.Module):
    def __init__(self, attention_func):
        super(TransformerCell, self).__init__()
        self.attention_func = attention_func

    # x is of size (batch, vector_length)
    # previous is of size (batch, sequence_length, vector_length)
    def forward(self, token, previous):
        queries = token.unsqueeze(1)
        keys = torch.cat([previous, token.unsqueeze(1)], 1) if previous is not None else token.unsqueeze(1)
        result = self.attention_func(queries, keys, keys, mask=None)[:,0] # batch_size, vector_length
        next_previous = keys
        return result, next_previous

# optimized multi-head Scaled Dot Product Attention with linear layers
# Note: Instead of splitting linear layers for each head, vector is split after linear layers applied
# Note: Different heads are treated as different batches when going through the SDPA
class CustomScaledDotProductAttention(nn.Module):
    def __init__(self, num_features, num_hidden, num_heads):
        super(CustomScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.query_layer = nn.Linear(num_features, num_hidden)
        self.key_layer = nn.Linear(num_features, num_hidden)
        self.value_layer = nn.Linear(num_features, num_hidden)
        self.sdpa = ScaledDotProductAttention()

    def forward(self, queries, keys, values, mask=None, return_distribution=False):
        b = queries.size(0)
        nq = queries.size(1)
        queries, keys, values = [
            layer(inputs).view(b, inputs.size(1), self.num_heads, -1)\
                         .transpose(1, 2).contiguous()\
                         .view(b*self.num_heads, inputs.size(1), -1)\
            for layer, inputs in zip((self.query_layer, self.key_layer, self.value_layer),
                                     (queries, keys, values))
        ]
        if mask is not None:
            mask = mask.view(mask.shape[0], 1, *mask.shape[1:])\
                       .expand(mask.shape[0], self.num_heads, *mask.shape[1:]).contiguous()\
                       .view(b*self.num_heads, *mask.shape[1:])
        return self.sdpa(queries, keys, values, mask=mask, return_distribution=False)\
            .view(b, self.num_heads, nq, -1)\
            .transpose(1, 2).contiguous()\
            .view(b, nq, -1)
