from gensim.models import Word2Vec
from data import get_data, Vectorizer
from train_word2vec import train_word2vec_model
import os
from models import Summarizer
from model_helpers import loss_function, error_function
from pytorch_helper import ModelManipulator, plot_learning_curves, plot_checkpoint
import torch
from utils import summarize, print_batch, visualize, produce_summary_files
import pdb


WORD2VEC_FILE = 'data/cnn_dataset/word2vec.model'
DATA_FILE = 'data/cnn_dataset/train_processed.data'
VAL_FILE = 'data/cnn_dataset/val_processed.data'
MODE = 'train'
CONTINUE_FROM_CHECKPOINT = True
CHECKPOINT_PATH = 'checkpoint'
MODEL_FILE = 'checkpoint/model.model'
POINTER_GEN = True
VISUALIZATION_FILE = 'graphs/attn_vis_data.json'
TRAINING_PLOTS_PATH = 'graphs'
MAX_TRAINING_STEPS = None

# hyperparameters
EMBEDDING_DIM = 128
LSTM_HIDDEN = 128
ATTN_HIDDEN = 128
WITH_COVERAGE = True
GAMMA = 1 # only matters if with_coverage = True
LEARNING_RATE = 1e-2
# INITIAL_ACCUMULATOR_VALUE = 0.1
BATCH_SIZE = 16
DECODING_BATCH_SIZE = 4
NUM_EPOCHS = 5
USE_CUDA = torch.cuda.is_available()
BEAM_SIZE = 4
AVERAGE_OVER = 1

if __name__ == '__main__':
    if not os.path.exists(WORD2VEC_FILE):
        train_word2vec_model(DATA_FILE, WORD2VEC_FILE, EMBEDDING_DIM)
    print("retreiving word2vec model from file")
    vectorizer = Vectorizer(Word2Vec.load(WORD2VEC_FILE))
    data = get_data(DATA_FILE, vectorizer, with_oov=POINTER_GEN)
    
    if MODE == 'train':
        val = get_data(VAL_FILE, vectorizer, with_oov=POINTER_GEN)
        
        if CONTINUE_FROM_CHECKPOINT:
            # check if all of the proper files exist
            model_file = os.path.join(CHECKPOINT_PATH, 'model.model')
            if not os.path.exists(model_file) or \
               not os.path.exists(os.path.join(CHECKPOINT_PATH, 'indices_iterator.pkl')) or \
               not os.path.exists(os.path.join(CHECKPOINT_PATH, 'iternum.txt')) or \
               not os.path.exists(os.path.join(CHECKPOINT_PATH, 'train_info.txt')) or \
               not os.path.exists(os.path.join(CHECKPOINT_PATH, 'val_info.txt')):
                print("Cannot continue from checkpoint because not all of the proper files exist; restarting.")
                CONTINUE_FROM_CHECKPOINT = False
            else:
                print("Loading from the last checkpoint; this expects the same datafile as before.")
        
        if CONTINUE_FROM_CHECKPOINT:
            model = torch.load(model_file)
        else:
            CONTINUE_FROM_CHECKPOINT = False
            start_index = vectorizer.word_indices['<start>']
            end_index = vectorizer.word_indices['<end>']
            model = Summarizer(
                vectorizer,
                start_index,
                end_index,
                lstm_hidden=LSTM_HIDDEN,
                attn_hidden=ATTN_HIDDEN,
                with_coverage=WITH_COVERAGE,
                gamma=GAMMA,
                with_pointer=POINTER_GEN
            )

        model = model if not USE_CUDA else model.cuda()
        
        if CONTINUE_FROM_CHECKPOINT:
            # TODO: change checkpoint so that the optimizer is loaded from the checkpoint
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
        model_manip = ModelManipulator(
            model,
            optimizer,
            loss_function,
            error_function
        )
        train_stats, val_stats = model_manip.train(
            data,
            BATCH_SIZE,
            NUM_EPOCHS,
            dataset_val=val,
            stats_every=10,
            verbose_every=10,
            checkpoint_every=10,
            checkpoint_path=CHECKPOINT_PATH,
            restart=not CONTINUE_FROM_CHECKPOINT,
            max_steps=MAX_TRAINING_STEPS,
        )
        if CONTINUE_FROM_CHECKPOINT:
            plot_checkpoint(
                CHECKPOINT_PATH,
                figure_name='plot',
                show=False,
                average_over=AVERAGE_OVER
            )
        else:
            plot_learning_curves(
                training_values=train_stats,
                validation_values=val_stats,
                figure_name=os.path.join(TRAINING_PLOTS_PATH, 'plot'),
                show=False,
                average_over=AVERAGE_OVER
            )

    elif MODE == 'eval':
        model = torch.load(MODEL_FILE)
        produce_summary_files(
            data,
            DECODING_BATCH_SIZE,
            vectorizer,
            model,
            'rouge',
            beam_size=BEAM_SIZE,
            max_num_batch=None
        )
        # TODO: run rouge
        # run_rouge()

    elif MODE == 'visualize':
        model = torch.load(MODEL_FILE)
        batch = data[:DECODING_BATCH_SIZE]
        results = summarize(batch, model, beam_size=BEAM_SIZE)
        print_batch(batch, results[0], vectorizer)
        visualize(VISUALIZATION_FILE, batch, results[0], vectorizer, 0)

