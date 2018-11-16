import os
from pytorch_helper import VariableLength
import pandas as pd
import torch

class SummarizationDataset:
    def __init__(self, df):
        self.df = df
        
    def read(self, i):
        text = self.prepare_text(self.df.text[i])
        summary = self.prepare_text(self.df.summary[i])
        return text, summary
    
    def prepare_text(self, text):
        return ['<start>'] + text + ['<end>']

    def text_iterator(self):
        return self.TextIterator(self)
    
    def __len__(self):
        return len(self.df)

    class TextIterator:
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            for i in range(len(self.dataset)):
                text, summary = self.dataset.read(i)
                yield text
                yield summary

class PreprocessedSummarizationDataset(VariableLength):
    def __init__(self, dataset, vectorizer, with_oov=False):
        self.dataset = dataset
        self.vectorizer = vectorizer
        self.with_oov = with_oov
        
    def get_raw_inputs(self, i):
        return self.dataset.read(i), None
    
    def prepare_inputs(self, v_args, nv_args, lengths):
        text, summary = v_args
        text_length_max, summary_length_max = lengths
        text, text_length, text_oov_indices = self.vectorizer.get_text_indices(text, text_length_max)
        summary, summary_length, _ = self.vectorizer.get_text_indices(summary, summary_length_max, oov_indices=text_oov_indices)
        return_dict = dict(text=text, text_length=text_length, summary=summary, summary_length=summary_length)
        if self.with_oov:
            return_dict['text_oov_indices'] = text_oov_indices
        return return_dict
    
    def __len__(self):
        return len(self.dataset)


class Vectorizer:
    def __init__(self, word2vec_model):
        self.word_indices = {}
        self.word_vectors = torch.zeros((len(word2vec_model.wv.vocab), word2vec_model.vector_size))
        self.words = []
        for i,k in enumerate(word2vec_model.wv.vocab):
            v = word2vec_model.wv[k]
            self.word_indices[k] = i
            self.word_vectors[i] = torch.tensor(v)
            self.words.append(k)
        
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
        word_list = []
        for i in text_indices:
            if i < len(self.words) and i >= 0:
                word = self.words[i]
            else:
                i_temp = len(oov_words)+len(self.words)-i-1 if oov_words is not None and i >= len(self.words) else -i-1
                if oov_words is None or i_temp >= len(oov_words):
                    word = 'oov'
                else:
                    word = oov_words[i_temp]
            word_list.append(word)
        return word_list

    def get_text_matrix(self, text_indices, max_length):
        if max_length < len(text_indices):
            raise Exception
        vectors = torch.zeros((max_length, len(self.word_vectors[0])), device=text_indices.device)
        for i,index in enumerate(text_indices):
            vectors[i,:] = torch.tensor(self.word_vectors[index])\
                           if index >= 0 and index < len(self.word_vectors) else torch.randn(len(self.word_vectors[0]))#torch.zeros(len(self.word_vectors[0]))
        return vectors.float(), torch.tensor(len(text_indices))

def get_data(data_file, vectorizer, with_oov=False):
    data = pd.read_json(data_file, lines=True, compression='gzip')
    print(len(data))
    return PreprocessedSummarizationDataset(SummarizationDataset(data), vectorizer, with_oov=with_oov)
