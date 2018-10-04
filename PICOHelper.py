import pandas as pd
import torch
from pytorch_helper import VariableLength
from utils import preprocess_text, get_text_indices, train_word2vec_model, get_indices_split, DataFrameDataset

class PICODataset(DataFrameDataset):
    def get_valid_rows(self):
        return (self.df.abstract == self.df.abstract) & (self.df.population == self.df.population) &\
               (self.df.intervention == self.df.intervention) & (self.df.outcome == self.df.outcome)

    def preprocess_pico_text(self, text):
        return ['<start>'] + preprocess_text(text) + ['<end>']
    
    def preprocess(self, i):
        if i not in set(self.indices):
            raise Exception
        abstract = self.preprocess_pico_text(self.df.abstract[i])
        P = self.preprocess_pico_text(self.df.population[i])
        I = self.preprocess_pico_text(self.df.intervention[i])
        O = self.preprocess_pico_text(self.df.outcome[i])
        return abstract, (P, I, O)

    def text_iterator(self):
        return self.TextIterator(self)

    class TextIterator:
        def __init__(self, pico):
            self.pico = pico

        def __iter__(self):
            for i in self.pico.indices:
                abstract, (P, I, O) = self.pico.preprocess(i)
                yield abstract
                yield P
                yield I
                yield O

# PICO Dataset Wrapper
class PICODataset_word2vec(VariableLength):
    def __init__(self, dataset, preprocess_model):
        self.dataset = dataset
        self.word_indices = {}
        self.word_vectors = torch.zeros((len(preprocess_model.wv.vocab), preprocess_model.vector_size))
        self.words = []
        for i,k in enumerate(preprocess_model.wv.vocab):
            v = preprocess_model.wv[k]
            self.word_indices[k] = i
            self.word_vectors[i] = torch.tensor(v)
            self.words.append(k)
            
    def get_raw_inputs(self, i):
        abstract, (P, I, O) = self.dataset.preprocess(self.dataset.indices[i])
        return (abstract, P, I, O), None
    
    def prepare_inputs(self, v_args, nv_args, lengths):
        abstract, P, I, O = v_args
        abstract_length_max, P_length_max, I_length_max, O_length_max = lengths
        abstract, abstract_length = get_text_indices(abstract, self.word_indices, abstract_length_max)
        P, P_length = get_text_indices(P, self.word_indices, P_length_max)
        I, I_length = get_text_indices(I, self.word_indices, I_length_max)
        O, O_length = get_text_indices(O, self.word_indices, O_length_max)
        return dict(abstract=abstract, abstract_length=abstract_length, P=P, P_length=P_length, I=I, I_length=I_length, O=O, O_length=O_length)
    
    def __len__(self):
        return len(self.dataset)
    
def get_pico_datasets():
    df = pd.read_csv("data/study_inclusion.csv")
    train_indices, dev_indices, test_indices = get_indices_split(len(df),.6,.2)
    print(len(train_indices), len(dev_indices), len(test_indices))
    pico_dataset_train = PICODataset(df, train_indices)
    word2vec_model = train_word2vec_model("data/word2vec_min5_pico.model", document_iterator=pico_dataset_train.text_iterator(), size=100, window=5, min_count=5, workers=4)
    pico_dataset_train_word2vec = PICODataset_word2vec(pico_dataset_train, word2vec_model)
    pico_dataset_dev_word2vec = PICODataset_word2vec(PICODataset(df, dev_indices), word2vec_model)
    pico_dataset_test_word2vec = PICODataset_word2vec(PICODataset(df, test_indices), word2vec_model)
    return pico_dataset_train_word2vec, pico_dataset_dev_word2vec, pico_dataset_test_word2vec