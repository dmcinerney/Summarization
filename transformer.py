import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, hidden_size, positional_encoding, attention_func):
        self.hidden_size = hidden_size
        self.positional_encoding = positional_encoding
        self.attention_func = attention_func
    
    def forward(self, text, text_length):
        mask = torch.arange(text.size(1)).expand(text.size()) < text_length.unsqueeze(1)
        
        return self.attention_func(x_pe, x_pe, x, mask=mask) # batch_size, sequence_length, vector_length

class TransformerCell(nn.Module):
    def 

class PositionalEncoding(nn.Module):
    pass
