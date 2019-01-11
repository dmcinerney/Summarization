import torch
from torch import nn
import math

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

# needs to be used if 
class TransformerCell(nn.Module):
    def __init__(self, attention_func):
        self.attention_func = attention_func

    # x is of size (batch, vector_length)
    # previous is of size (batch, sequence_length, vector_length)
    def forward(self, token, previous):
        queries = token.unsqueeze(1)
        keys = torch.cat([previous, token.unsqueeze(1)], 1)
        result = self.attention_func(queries, keys, keys, mask=None)[0] # batch_size, vector_length
        next_previous = keys
        return result, next_previous