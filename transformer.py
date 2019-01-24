import torch
from torch import nn
import math
from attention import ScaledDotProductAttention
import torch.nn.functional as F
import pdb

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

    def forward(self, x, length, return_distribution=False):
        arangement = torch.arange(x.size(1), device=x.device)
        mask = arangement.unsqueeze(0) < length.unsqueeze(1) # batch_size, sequence_length
        mask = mask.unsqueeze(1) # batch_size, 1, sequence_length
        if self.unidirectional:
            unimask = arangement.unsqueeze(0) <= arangement.unsqueeze(1) # sequence_length, sequence_length
            mask = mask*unimask.unsqueeze(0) # batch_size, sequence_length, sequence_length
        return self.attention_func(x, x, x, mask=mask, return_distribution=return_distribution)
            # batch_size, sequence_length, vector_length

class TransformerCell(nn.Module):
    def __init__(self, attention_func):
        super(TransformerCell, self).__init__()
        self.attention_func = attention_func

    # x is of size (batch, vector_length)
    # previous is of size (batch, sequence_length, vector_length)
    def forward(self, token, previous, return_distribution=False):
        queries = token.unsqueeze(1)
        keys = next_previous = torch.cat([previous, token.unsqueeze(1)], 1) if previous is not None else token.unsqueeze(1)
        if return_distribution:
            result, distribution = self.attention_func(queries, keys, keys, mask=None, return_distribution=True)
                # (batch_size, 1, vector_length), (batch_size, 1, sequence_length)
            return result[:,0], next_previous, distribution[:,0]
        else:
            result = self.attention_func(queries, keys, keys, mask=None, return_distribution=False)
                # batch_size, 1, vector_length
            return result[:,0], next_previous

# optimized multi-head Scaled Dot Product Attention with linear layers
# Note: Instead of splitting linear layers for each head, vector is split after linear layers applied
# Note: Different heads are treated as different batches when going through the SDPA
class CustomScaledDotProductAttention(nn.Module):
    def __init__(self, num_features, num_hidden, num_heads, dropout=None):
        super(CustomScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.query_layer = nn.Linear(num_features, num_hidden)
        self.key_layer = nn.Linear(num_features, num_hidden)
        self.value_layer = nn.Linear(num_features, num_hidden)
        self.sdpa = ScaledDotProductAttention()
        self.final_layer = nn.Linear(num_hidden, num_hidden)
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

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
        results = self.sdpa(queries, keys, values, mask=mask, return_distribution=return_distribution, dropout=self.dropout)
        if return_distribution:
            results, distribution = results
        results = results.view(b, self.num_heads, nq, -1)\
                         .transpose(1, 2).contiguous()\
                         .view(b, nq, -1)
        results = self.final_layer(results)
        if return_distribution:
            distribution = distribution.view(b, self.num_heads, nq, -1)\
                                       .transpose(1, 2).contiguous()
            return results, distribution
        else:
            return results

# Takne from http://nlp.seas.harvard.edu/2018/04/03/attention.html
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class CustomTransformer(nn.Module):
    def __init__(self, num_features, num_heads, unidirectional=False, dropout=None):
        super(CustomTransformer, self).__init__()
        if num_features % num_heads != 0:
            raise Exception
        self.transformer = Transformer(CustomScaledDotProductAttention(num_features, num_features, num_heads, dropout=dropout), unidirectional=unidirectional)
        self.normalize1 = LayerNorm(num_features)
        self.linear1 = nn.Linear(num_features, num_features*4)
        self.linear2 = nn.Linear(num_features*4, num_features)
        self.normalize2 = LayerNorm(num_features)

    def forward(self, x, length):
        x2, distribution = self.transformer(x, length, return_distribution=True)
        x2 = self.normalize1(x2 + x)
        results = self.linear2(F.relu(self.linear1(x2))) # batch_size, sequence_length, vector_length
        results = self.normalize2(results + x2)
        return results, distribution

class CustomTransformerCell(nn.Module):
    def __init__(self, num_features, num_heads, dropout=None):
        super(CustomTransformerCell, self).__init__()
        if num_features % num_heads != 0:
            raise Exception
        self.transformer_cell = TransformerCell(CustomScaledDotProductAttention(num_features, num_features, num_heads, dropout=dropout))
        self.normalize1 = LayerNorm(num_features)
        self.linear1 = nn.Linear(num_features, num_features*4)
        self.linear2 = nn.Linear(num_features*4, num_features)
        self.normalize2 = LayerNorm(num_features)

    def forward(self, token, previous):
        token2, next_previous, distribution = self.transformer_cell(token, previous, return_distribution=True)
        token2 = self.normalize1(token2 + token)
        new_token = self.linear2(F.relu(self.linear1(token2)))
        new_token = self.normalize2(new_token + token2)
        return new_token, next_previous, distribution
