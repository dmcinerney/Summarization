from data import SummarizationDataset
import pandas as pd
from gensim.models import Word2Vec

def train_word2vec_model(data_file, word2vec_file, embedding_dim):
    print("reading data file")
    data = pd.read_json(data_file, lines=True, compression='gzip')
    document_iterator = SummarizationDataset(data).text_iterator()
    print("training word2vec model")
    word2vec_model = Word2Vec(document_iterator, size=embedding_dim, window=5, min_count=5, workers=4)
    word2vec_model.save(word2vec_file)

DATA_FILE = 'data/data/train_processed.data'
WORD2VEC_FILE = 'data/data/word2vec_2.model'
EMBEDDING_DIM = 128

if __name__ == '__main__':
    train_word2vec_model(DATA_FILE, WORD2VEC_FILE, EMBEDDING_DIM)
