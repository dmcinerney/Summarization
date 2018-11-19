import torch

WORD2VEC_FILE = 'data/cnn_dataset/word2vec.model'
DATA_FILE = 'data/cnn_dataset/train_processed.data'
VAL_FILE = 'data/cnn_dataset/val_processed.data'
MODE = 'visualize'
CONTINUE_FROM_CHECKPOINT = True
CHECKPOINT_PATH = 'checkpoint'
MODEL_FILE = 'checkpoint4/model.model'
POINTER_GEN = True
VISUALIZATION_FILE = 'graphs/attn_vis_data.json'
TRAINING_PLOTS_PATH = 'graphs'
MAX_TRAINING_STEPS = None

# hyperparameters
EMBEDDING_DIM = 64
LSTM_HIDDEN = 64
ATTN_HIDDEN = 64
WITH_COVERAGE = True
GAMMA = 1 # only matters if with_coverage = True
LEARNING_RATE = .015
INITIAL_ACCUMULATOR_VALUE = 0.1
BATCH_SIZE = 16
DECODING_BATCH_SIZE = 4
NUM_EPOCHS = 5
USE_CUDA = torch.cuda.is_available()
BEAM_SIZE = 4
AVERAGE_OVER = 1
WEIGHT_INIT_MEAN=0.02
WEIGHT_INIT_STD=1e-4
MAX_GRAD_NORM=2.0
MAX_TEXT_LENGTH = 400
MAX_SUMMARY_LENGTH = 100

params = dir()
def save_params(filename):
    with open(filename, 'w') as file:
        for param in params:
            if not param.startswith('__') and param != 'torch':
                file.write(param+' = '+str(eval(param))+'\n')
