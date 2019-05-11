import os
from pytorch_helper import VariableLength
import pandas as pd
import torch
from torch import nn
from datasets import SummarizationDataset, PreprocessedSummarizationDataset, HierarchicalDataset
from model_helpers import init_weights_normal
import pdb


class Vectorizer(nn.Module):
    def __init__(self):
        super(Vectorizer, self).__init__()

    def __len__(self):
        return len(self.words)

    def get_text_indices(self, text, max_length, oov_indices=None):
        if max_length < len(text):
            raise Exception
        oov_indices = {} if oov_indices is None else dict(oov_indices)
        indices = torch.zeros((max_length))
        for i,token in enumerate(text):
            if token in self.word_indices:
                indices[i] = torch.tensor(self.word_indices[token]).long()
            else:
                if token not in oov_indices.keys():
                    oov_indices[token] = len(oov_indices)
                indices[i] = torch.tensor(-1-oov_indices[token]).long()
        return indices.int(), torch.tensor(len(text)), oov_indices

    def get_index_words(self, text_indices, oov_words=None):
        if (text_indices >= len(self.word_indices)).any():
            raise Exception
        word_list = []
        for i in text_indices:
            if i >= 0:
                word = self.words[i]
            else:
                if oov_words is None or i < -len(oov_words):
                    word = 'oov'
                else:
                    word = oov_words[-i-1]
            word_list.append(word)
        return word_list

    def forward(self, indices, lengths):
        raise NotImplementedError

    @property
    def vocab_size(self):
        raise NotImplementedError

    @property
    def vector_size(self):
        raise NotImplementedError

class Word2VecVectorizer(Vectorizer):
    def __init__(self, word2vec_model):
        super(Word2VecVectorizer, self).__init__()
        self.word_indices = {}
        self.word_vectors = torch.zeros((len(word2vec_model.wv.vocab), word2vec_model.vector_size))
        self.words = []
        for i,k in enumerate(word2vec_model.wv.vocab):
            v = word2vec_model.wv[k]
            self.word_indices[k] = i
            self.word_vectors[i] = torch.tensor(v)
            self.words.append(k)

    def forward(self, text, lengths):
        return torch.cat([self.get_text_matrix(example[:lengths[i]], text.size(1))[0].unsqueeze(0) for i,example in enumerate(text)], 0)

    def get_text_matrix(self, text_indices, max_length):
        if max_length < len(text_indices):
            raise Exception
        vectors = torch.zeros((max_length, len(self.word_vectors[0])), device=text_indices.device)
        for i,index in enumerate(text_indices):
            vectors[i,:] = torch.tensor(self.word_vectors[index])\
                           if index >= 0 and index < len(self.word_vectors) else torch.randn(len(self.word_vectors[0]))#torch.zeros(len(self.word_vectors[0]))
        return vectors.float(), torch.tensor(len(text_indices))

    @property
    def vocab_size(self):
        return len(self.words)

    @property
    def vector_size(self):
        return len(self.word_vectors[0])

class TrainableVectorizer(Vectorizer):
    def __init__(self, dictionary, vector_size):
        super(TrainableVectorizer, self).__init__()
        self.word_indices = dictionary.token2id
        self.words = {v:k for k,v in self.word_indices.items()}#dictionary.id2token
        self._vector_size = vector_size
        self.embedding = nn.Embedding(self.vocab_size+1, vector_size)
        init_weights_normal(self.embedding.weight)

    def forward(self, indices, lengths):
        padding = torch.arange(indices.size(1), device=indices.device).expand(*indices.size()) >= lengths.unsqueeze(1)
        indices_mod = indices.clone()
        indices_mod[padding] = 0.
        indices_mod[(indices_mod < 0) | (indices_mod >= self.vocab_size)] = self.vocab_size
        return self.embedding(indices_mod.long())

    @property
    def vocab_size(self):
        return len(self.words)

    @property
    def vector_size(self):
        return self._vector_size

def get_data(data_file, vectorizer, with_oov=False, aspect_file=None, hierarchical=False):
    print('reading data from '+data_file)
    data = pd.read_json(data_file, lines=True, compression='gzip')
    print('data read, length is %i' % len(data))
    Dataset = PreprocessedSummarizationDataset if not hierarchical else HierarchicalDataset
    return Dataset(SummarizationDataset(data, aspect_file=aspect_file), vectorizer, with_oov=with_oov)
