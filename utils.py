from gensim.models import Word2Vec
import os.path
import spacy, re
import torch
import numpy as np
import json
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = [t.text.lower() for t in nlp(text)]
    text = ['qqq' if any(char.isdigit() for char in word) else word for word in text]
    return text

def get_text_indices(text, word_indices, max_length, oov_indices=None):
    if max_length < len(text):
        raise Exception
    oov_indices = {} if oov_indices is None else dict(oov_indices)
    indices = torch.zeros((max_length))
    for i,token in enumerate(text):
        if token in word_indices:
            indices[i] = torch.tensor(word_indices[token]).long()
        else:
            if token not in oov_indices.keys():
                oov_indices[token] = len(oov_indices)
            indices[i] = torch.tensor(-1-oov_indices[token]).long()
    return indices.int(), torch.tensor(len(text)), oov_indices

def get_index_words(text_indices, words, oov_words=None):
    word_list = []
    for i in text_indices:
        if i < len(words) and i >= 0:
            word = words[i]
        else:
            i_temp = len(oov_words)+len(words)-i-1 if oov_words is not None and i >= len(words) else -i-1
            if oov_words is None or i_temp >= len(oov_words):
                word = 'oov'
            else:
                word = oov_words[i_temp]
        word_list.append(word)
    return word_list

def get_text_matrix(text_indices, word_vectors, max_length):
    if max_length < len(text_indices):
        raise Exception
    vectors = torch.zeros((max_length, len(word_vectors[0])), device=text_indices.device)
    for i,index in enumerate(text_indices):
        vectors[i,:] = torch.tensor(word_vectors[index])\
                       if index >= 0 and index < len(word_vectors) else torch.zeros(len(word_vectors[0]))#torch.randn(len(word_vectors[0]))
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
        
def produce_attention_visualization_file(filename, text, decoded_summary, reference_summary, attentions, p_gens):
    data = {'article_lst': text,
            'decoded_lst': decoded_summary,
            'abstract_str': reference_summary,
            'attn_dists': attentions,
            'p_gens': p_gens}
    
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)