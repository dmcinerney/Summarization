import pandas as pd
from pytorch_helper import VariableLength
from utils import preprocess_text, train_word2vec_model, DataFrameDataset, Preprocessor
import torch

class NewsroomDataset(DataFrameDataset):
    def get_valid_rows(self):
        return (self.df.text == self.df.text) & (self.df.summary == self.df.summary)
    
    def preprocess_news_text(self, text):
        return ['<start>'] + text + ['<end>']
    
    def preprocess(self, i):
        if i not in set(self.indices):
            raise Exception
        text = self.preprocess_news_text(self.df.text[i])
        summary = self.preprocess_news_text(self.df.summary[i])
        return text, summary

    def text_iterator(self):
        return self.TextIterator(self)

    class TextIterator:
        def __init__(self, newsroom):
            self.newsroom = newsroom

        def __iter__(self):
            for i in self.newsroom.indices:
                text, summary = self.newsroom.preprocess(i)
                yield text
                yield summary

# Newsroom Dataset Wrapper
class NewsroomDataset_word2vec(VariableLength):
    def __init__(self, dataset, preprocessor, with_oov=False):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.with_oov = with_oov
        
    def get_raw_inputs(self, i):
        return self.dataset.preprocess(self.dataset.indices[i]), None
    
    def prepare_inputs(self, v_args, nv_args, lengths):
        text, summary = v_args
        text_length_max, summary_length_max = lengths
        text, text_length, text_oov_indices = self.preprocessor.get_text_indices(text, text_length_max)
        summary, summary_length, _ = self.preprocessor.get_text_indices(summary, summary_length_max, oov_indices=text_oov_indices)
        return_dict = dict(text=text, text_length=text_length, summary=summary, summary_length=summary_length)
        if self.with_oov:
            return_dict['text_oov_indices'] = text_oov_indices
        return return_dict
    
    def __len__(self):
        return len(self.dataset)

# this is temporary
from utils import get_indices_split

def get_newsroom_datasets(with_oov=False):
#     train = pd.read_json('data/train.data', lines=True, compression='gzip')
#     dev = pd.read_json('data/dev.data', lines=True, compression='gzip')
#     test = pd.read_json('data/test.data', lines=True, compression='gzip')
#     print(len(train), len(dev), len(test))
#     newsroom_dataset_train = NewsroomDataset(train)
#     word2vec_model = train_word2vec_model("data/word2vec_min5_newsroom.model", document_iterator=newsroom_dataset_train.text_iterator(), size=100, window=5, min_count=5, workers=4)
#     preprocessor = Preprocessor(word2vec_model)
#     newsroom_dataset_train_word2vec = NewsroomDataset_word2vec(newsroom_dataset_train, preprocessor, 73733)
#     newsroom_dataset_dev_word2vec = NewsroomDataset_word2vec(NewsroomDataset(dev), preprocessor, 73733)
#     newsroom_dataset_test_word2vec = NewsroomDataset_word2vec(NewsroomDataset(test), preprocessor, 73733)
    
    train = pd.read_json('data/train_trimmed.data', lines=True, compression='gzip')
    dev = pd.read_json('data/dev_trimmed.data', lines=True, compression='gzip')
#     train_indices, _, _ = get_indices_split(300,.9,.05)
#     train_indices, _, _ = get_indices_split(len(train),.9,.05)
    dev_indices, test_indices, _ = get_indices_split(len(dev),.45,.45)
#     train_indices, dev_indices, test_indices = get_indices_split(1000,.6,.2)
    print(train.shape[0], len(dev_indices), len(test_indices))
    newsroom_dataset_train = NewsroomDataset(train)
    size, min_count = 100, 5
    word2vec_model = train_word2vec_model("models/word2vec_%id_min%i_newsroom.model" % (size, min_count), document_iterator=newsroom_dataset_train.text_iterator(), size=size, window=5, min_count=min_count, workers=4)
    preprocessor = Preprocessor(word2vec_model)
    newsroom_dataset_train_word2vec = NewsroomDataset_word2vec(newsroom_dataset_train, preprocessor, with_oov=with_oov)
    newsroom_dataset_dev_word2vec = NewsroomDataset_word2vec(NewsroomDataset(dev, dev_indices), preprocessor, with_oov=with_oov)
    newsroom_dataset_test_word2vec = NewsroomDataset_word2vec(NewsroomDataset(dev, test_indices), preprocessor, with_oov=with_oov)
    return newsroom_dataset_train_word2vec, newsroom_dataset_dev_word2vec, newsroom_dataset_test_word2vec, preprocessor