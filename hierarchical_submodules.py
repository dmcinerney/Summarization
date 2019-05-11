from submodules import *

# linear, activation, linear, softmax, sum where
# input is:
#     the hidden states from the TextEncoder, the current state from the SummaryDecoder
# and outputs are:
#     context_vector (a weighted sum of the encoder hidden states according to the attention)
#     attention (a softmax of a vector the length of the number of hidden states)
class HierarchicalContextVectorNN(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(HierarchicalContextVectorNN, self).__init__()
#         self.linear1 = nn.Linear(num_inputs, num_hidden)
#         self.linear2 = nn.Linear(num_hidden, 1)
        self.word_additive_attention = AdditiveAttention(num_inputs, num_hidden)
        self.sentence_additive_attention = AdditiveAttention(num_inputs, num_hidden)

    def forward(self, text_states, sentence_states, text_length, summary_current_state, coverage):
        batch_size, num_sentences, num_words, _ = text_states.shape
        coverages = coverage.unsqueeze(2) # batch_size, sequence_length, 1
        flattened_coverages = coverages.view(batch_size, num_sentences, num_words, 1)\
                                       .view(batch_size*num_sentences, num_words, 1)
        flattened_text_states = text_states.view(batch_size*num_sentences, num_words, -1)
        
        mask = torch.arange(num_words, device=text_states.device) < text_length.unsqueeze(2)
        
        queries = summary_current_state.expand(num_sentences, *summary_current_state.shape).transpose(0, 1)\
                                       .contiguous()\
                                       .view(batch_size*num_sentences, 1, -1)
        keys = torch.cat([flattened_text_states, flattened_coverages], 2) # batch_size, sequence_length, ts_vector_length+1
        values = flattened_text_states
            # batch_size, sequence_length
        context_vector, word_attention = self.word_additive_attention(queries, keys, values, mask=mask.view(batch_size*num_sentences, 1, num_words), return_distribution=True)
        set_to_zero = mask.view(batch_size*num_sentences, num_words).sum(1) == 0
        context_vector = context_vector.masked_fill(set_to_zero.unsqueeze(1).unsqueeze(2), 0)
        word_attention = word_attention.masked_fill(set_to_zero.unsqueeze(1).unsqueeze(2), 0)
        sentence_context_vectors = context_vector.view(batch_size, num_sentences, -1)
        
        flattened_coverages = coverages.view(batch_size, num_sentences, num_words, 1)\
                                       .sum(2)
        
        queries = summary_current_state
        keys = torch.cat([sentence_states, flattened_coverages], 2)
        values = sentence_context_vectors
        context_vector1, sentence_attention = self.sentence_additive_attention(queries, keys, values, mask=(text_length != 0).unsqueeze(1), return_distribution=True)
        values = sentence_states
        context_vector2, sentence_attention = self.sentence_additive_attention(queries, keys, values, mask=(text_length != 0).unsqueeze(1), return_distribution=True)
        attention = (word_attention.view(batch_size, 1, num_sentences, num_words)*sentence_attention.unsqueeze(3))\
                    .view(batch_size, 1, num_sentences*num_words)
        return torch.cat([context_vector1, context_vector2], 2), attention

class ReduceContextVector(nn.Module):
    def __init__(self, num_inputs):
        super(ReduceContextVector, self).__init__()
        self.linear = nn.Linear(num_inputs, num_inputs//2)
        
    def forward(self, context_vector):
        return self.linear(context_vector)