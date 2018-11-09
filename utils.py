from gensim.models import Word2Vec
import os.path
import spacy, re
import torch
import numpy as np
import json
from pyrouge import Rouge155
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
        
class Preprocessor:
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
                           if index >= 0 and index < len(self.word_vectors) else torch.zeros(len(self.word_vectors[0]))#torch.randn(len(word_vectors[0]))
        return vectors.float(), torch.tensor(len(text_indices))
        
def produce_attention_visualization_file(filename, text, reference_summary, decoded_summary, attentions, p_gens):
    data = {'article_lst': text,
            'decoded_lst': decoded_summary,
            'abstract_str': reference_summary,
            'attn_dists': attentions,
            'p_gens': p_gens}
    
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
        
def summarize(batch, model, beam_size=1):
    if model.with_pointer:
        results = model(batch['text'].cuda(), batch['text_length'].cuda(), batch['text_oov_indices'], beam_size=beam_size)
    else:
        results = model(batch['text'].cuda(), batch['text_length'].cuda(), beam_size=beam_size)
    return results
    
def get_text_triplets(batch, summary_info, preprocessor, pointer_gen=False):
    text_triplets = []
    for i in range(len(summary_info[0])):
        summary_indices, summary_length = summary_info[0][i], summary_info[1][i]
        r_summary_indices, r_summary_length = batch['summary'][i].numpy(), batch['summary_length'][i].numpy()
        text_indices, text_length = batch['text'][i].numpy(), batch['text_length'][i].numpy()
        oov_words = {v:k for k,v in batch['text_oov_indices'][i].items()} if pointer_gen is True else None

        text = preprocessor.get_index_words(text_indices[:text_length], oov_words=oov_words)
        reference_summary = preprocessor.get_index_words(r_summary_indices[:r_summary_length], oov_words=oov_words)
        decoded_summary = preprocessor.get_index_words(summary_indices[:summary_length], oov_words=oov_words)
        
        text_triplets.append((text,reference_summary,decoded_summary))
    return text_triplets

def produce_batch_summary_files(batch, model, path, beam_size=1, start_index=0):
    results = summarize(batch, model, beam_size=beam_size)
    text_triplets = get_text_triplets(batch, results[0], model.preprocessor)
    i = start_index
    for text,reference,decoded in text_triplets:
        with open(os.path.join(path,"articles/article"+str(i)+"_text.txt"), "w") as textfile:
            textfile.write(" ".join(text[1:-1]))
        with open(os.path.join(path,"reference/article"+str(i)+"_reference.txt"), "w") as referencefile:
            referencefile.write(" ".join(reference[1:-1]))
        with open(os.path.join(path,"system/article"+str(i)+"_system.txt"), "w") as decodedfile:
            decodedfile.write(" ".join(decoded[1:-1]))
        i += 1
    return i
    
def produce_summary_files(dataloader, model, path, beam_size=1, max_num_batch=None):
    start_index = 0
    for i,batch in enumerate(dataloader):
        start_index = produce_batch_summary_files(batch, model, path, beam_size=beam_size, start_index=start_index)
        print(i)
        if max_num_batch is not None and (i+1) >= max_num_batch:
            break