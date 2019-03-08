import torch

#ASPECT_FILE = 'data/pico_dataset/aspects.txt'
ASPECT_FILE = None
WORD2VEC_FILE = 'data/cnn_dataset/word2vec128.model'
DICTIONARY_FILE = 'data/cnn_dataset/dictionary.model'
DATA_FILE = 'data/cnn_dataset/train_processed.data'
VAL_FILE = 'data/cnn_dataset/val_processed.data'
#WORD2VEC_FILE = 'data/pico_dataset/word2vec.model'
#DATA_FILE = 'data/pico_dataset/train_processed.data'
#VAL_FILE = 'data/pico_dataset/dev_processed.data'
MODE = 'eval'
CONTINUE_FROM_CHECKPOINT = True
CHECKPOINT_PATH = 'checkpoints/_'
MODEL_FILE = 'checkpoints/_/model_state.pkl'
VISUALIZATION_FILE = 'checkpoints/_/attn_vis_data.json'
TRAINING_PLOTS_PATH = 'checkpoints/_/'
TEXT_PATH = 'checkpoints/_'
NEW_EPOCH = False
DETECT_ANOMALY = False
MAX_TRAINING_STEPS = 100000
POINTER_GEN = True
USE_TRANSFORMER = False
PRETRAINED_WORD_VECTORS = False

# hyperparameters
EMBEDDING_DIM = 128
LSTM_HIDDEN = 256
ATTN_HIDDEN = 256*2
# EMBEDDING_DIM = 64
# LSTM_HIDDEN = 64
# ATTN_HIDDEN = 64*2
NUM_TRANSFORMER_HEADS = 8
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = .1
P_GEN = None # if none, p_gen is calculated as usual, else p_gen is manually set to P_GEN
EPSILON = .0000000 # mostly just used when P_GEN is set to 0
WITH_COVERAGE = True
GAMMA = 1 # only matters if with_coverage = True
LEARNING_RATE = .15
INITIAL_ACCUMULATOR_VALUE = 0.1
BATCH_SIZE = 16
DECODING_BATCH_SIZE = 4
NUM_EPOCHS = 50
DEVICE = 'cuda:1'
BEAM_SIZE = 4
AVERAGE_OVER = 30
WEIGHT_INIT_MAG = 0.02
WEIGHT_INIT_STD = 1e-4
MAX_GRAD_NORM = 2.0
MAX_TEXT_LENGTH = 400
MAX_SUMMARY_LENGTH = 100

param_names = [param for param in dir() if not param.startswith('__') and param != 'torch']
def save_params(filename):
    with open(filename, 'w') as file:
        for param in param_names:
            file.write(param+' = '+str(eval(param))+'\n')
