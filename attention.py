import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.jit import ScriptModule, script_method

class Attention(ScriptModule):
    def __init__(self):
        super(Attention, self).__init__()
    
    # keys and values - have size (batch_size, sequence_length, vector_length)
    # queries - have size (batch_size, num_queries, vector_length)
    # Note: batch_sizes and squence_lengths all have to be equal,
    #   vector_lengths can vary between keys and values,
    #   vector_lengths can vary between keys and queries depending on type of attention
    # mask - needs a mask if, for example, the sequence lengths vary through the batch or
    #   certain queries do not attend over the whole sequence,
    #   will be a byte tensor of size (batch_size, num_queries, sequence_length) with ones on all valid positions
    @script_method
    def forward(self, queries, keys, values, mask):
        scores = self.scores(queries, keys) # batch_size, num_queries, sequence_length
        attention_dist = torch.exp(scores)
        attention_dist = attention_dist*mask.float()
        attention_dist = attention_dist/attention_dist.sum(2, keepdim=True)
        final_vector = (attention_dist.unsqueeze(3)*values.unsqueeze(1)).sum(2) # batch_size, num_queries, vector_length
        return final_vector, attention_dist
    
#     @script_method
#     def scores(self, queries, keys):
#         raise NotImplementedError

class AdditiveAttention(Attention):
    def __init__(self, input_size, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    @script_method
    def scores(self, queries, keys):
        b, n_q, q_v = queries.size()
        _, s, k_v = keys.size()
        inputs = torch.cat([queries.unsqueeze(2).expand(b, n_q, s, q_v), keys.unsqueeze(1).expand(b, n_q, s, k_v)], 3)
        scores = self.linear2(torch.tanh(self.linear1(inputs))) # batch_size, num_queries, sequence_length, 1
        return scores.squeeze(3)

class ScaledDotProductAttention(Attention):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    @script_method
    def scores(self, queries, keys):
        # queries (batch_size, num_queries, q_vec_length)
        # keys (batch_size, sequence_length, k_vec_length)
        return torch.bmm(queries, keys.transpose(1,2)) # (batch_size, num_queries, sequence_length)

# non-optimized multi-headed attention
class MultiHeadedAttention:
    def __init__(self, attention_object_generator, num_heads):
        super(MultiHeadedAttention, self).__init__()
        self.attention_heads = nn.ModuleList([attention_object_generator() for _ in range(num_heads)])

    def forward(self, *args, **kwargs):
        return torch.cat([attention_head(*args, **kwargs) for attention_head in self.attention_heads], 2)
