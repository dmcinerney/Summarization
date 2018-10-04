from gensim.models import Word2Vec
import os.path
import spacy, re
import torch
import numpy as np
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

def preprocess_text(text) :
    text = re.sub(r'\s+', ' ', text.strip())
    text = [t.text.lower() for t in nlp(text)]
    text = ['qqq' if any(char.isdigit() for char in word) else word for word in text]
    return text

def get_text_indices(text, word_indices, max_length):
    if max_length < len(text):
        raise Exception
    indices = torch.zeros((max_length))
    for i,token in enumerate(text):
        indices[i] = torch.tensor(word_indices[token])\
                     if token in word_indices else -1
    return indices.int(), torch.tensor(len(text))

def get_index_words(text_indices, words):
    word_list = []
    for i in text_indices:
        word_list.append(words[i])
    return word_list

def get_text_matrix(text_indices, word_vectors, max_length):
    if max_length < len(text_indices):
        raise Exception
    vectors = torch.zeros((max_length, len(word_vectors[0])), device=text_indices.device)
    for i,index in enumerate(text_indices):
        vectors[i,:] = torch.tensor(word_vectors[index])\
                       if index > 0 else torch.randn(len(word_vectors[0]))
    return vectors.float(), torch.tensor(len(text_indices))
            
def train_word2vec_model(filename, document_iterator=None, force_reload=False, **kwargs):
    if force_reload or not os.path.isfile(filename):
        print("training word2vec model")
        word2vec_model = Word2Vec(document_iterator, **kwargs)
        word2vec_model.save(filename)
        return word2vec_model
    else:
        print("retrieving word2vec model from file")
        return Word2Vec.load(filename)
    
def get_indices_split(length, train_proportion, dev_proportion):
    train_length = int(train_proportion*length)
    dev_length = int(dev_proportion*length)
    if train_length+dev_length >= length:
        raise Exception
    indices = np.random.permutation(length)
    return indices[:train_length], indices[train_length:train_length+dev_length], indices[train_length+dev_length:]

class DataFrameDataset:
    def __init__(self, df, indices=None):
        self.df = df
        self.indices = self.df[self.get_valid_rows()].index
        if indices is not None:
            self.indices = list(set(self.indices).intersection(set(indices)))
        self.indices = np.array(self.indices)
    
    def __len__(self):
        return len(self.indices)

    def preprocess(self, i):
        raise NotImplementedError
        
    def get_valid_rows(self):
        raise NotImplementedError
        
class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)
        self.indices = torch.tensor(self.indices)
        
    def __iter__(self):
        for i in range(len(self.indices)//self.batch_size):
            offset = int(i*self.batch_size)
            yield self.dataset[self.indices[offset:offset+self.batch_size]]
        