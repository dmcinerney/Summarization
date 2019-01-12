import torch

#ASPECT_FILE = 'data/pico_dataset/aspects.txt'
ASPECT_FILE = None
WORD2VEC_FILE = 'data/cnn_dataset/word2vec64.model'
# DICTIONARY_FILE = 'data/cnn_dataset/dictionary.model'
DATA_FILE = 'data/cnn_dataset/train_processed.data'
VAL_FILE = 'data/cnn_dataset/val_processed.data'
#WORD2VEC_FILE = 'data/pico_dataset/word2vec.model'
#DATA_FILE = 'data/pico_dataset/train_processed.data'
#VAL_FILE = 'data/pico_dataset/dev_processed.data'
MODE = 'train'
CONTINUE_FROM_CHECKPOINT = True
CHECKPOINT_PATH = 'checkpoint'
MODEL_FILE = 'checkpoint/model.model'
POINTER_GEN = True
VISUALIZATION_FILE = 'checkpoint/attn_vis_data.json'
TRAINING_PLOTS_PATH = 'checkpoint'
MAX_TRAINING_STEPS = None

# hyperparameters
# EMBEDDING_DIM = 128
# LSTM_HIDDEN = 256
# ATTN_HIDDEN = 256*2
EMBEDDING_DIM = 64
LSTM_HIDDEN = 64
ATTN_HIDDEN = 64
NUM_TRANSFORMER_HEADS = 4
WITH_COVERAGE = True
GAMMA = 1 # only matters if with_coverage = True
LEARNING_RATE = .15
INITIAL_ACCUMULATOR_VALUE = 0.1
BATCH_SIZE = 16
DECODING_BATCH_SIZE = 4
NUM_EPOCHS = 10
USE_CUDA = torch.cuda.is_available()
BEAM_SIZE = 4
AVERAGE_OVER = 1
WEIGHT_INIT_MAG=0.02
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
