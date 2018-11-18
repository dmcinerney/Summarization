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
import parameters as p
from model_helpers import clip_grad_norm

if __name__ == '__main__':
    if not os.path.exists(p.WORD2VEC_FILE):
        train_word2vec_model(p.DATA_FILE, p.WORD2VEC_FILE, p.EMBEDDING_DIM)
    print("retreiving word2vec model from file")
    vectorizer = Vectorizer(Word2Vec.load(p.WORD2VEC_FILE))
    data = get_data(p.DATA_FILE, vectorizer, with_oov=p.POINTER_GEN)

    if p.MODE == 'train':
        val = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN)

        if p.CONTINUE_FROM_CHECKPOINT:
            # check if all of the proper files exist
            model_file = os.path.join(p.CHECKPOINT_PATH, 'model.model')
            if not os.path.exists(model_file) or \
               not os.path.exists(os.path.join(p.CHECKPOINT_PATH, 'indices_iterator.pkl')) or \
               not os.path.exists(os.path.join(p.CHECKPOINT_PATH, 'iternum.txt')) or \
               not os.path.exists(os.path.join(p.CHECKPOINT_PATH, 'train_info.txt')) or \
               not os.path.exists(os.path.join(p.CHECKPOINT_PATH, 'val_info.txt')):
                print("Cannot continue from checkpoint because not all of the proper files exist; restarting.")
                p.CONTINUE_FROM_CHECKPOINT = False
            else:
                print("Loading from the last checkpoint; this expects the same datafile as before.")

        if p.CONTINUE_FROM_CHECKPOINT:
            model = torch.load(model_file)
        else:
            p.CONTINUE_FROM_CHECKPOINT = False
            start_index = vectorizer.word_indices['<start>']
            end_index = vectorizer.word_indices['<end>']
            model = Summarizer(
                vectorizer,
                start_index,
                end_index,
                lstm_hidden=p.LSTM_HIDDEN,
                attn_hidden=p.ATTN_HIDDEN,
                with_coverage=p.WITH_COVERAGE,
                gamma=p.GAMMA,
                with_pointer=p.POINTER_GEN
            )

        model = model if not p.USE_CUDA else model.cuda()

#         optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=p.LEARNING_RATE,
            initial_accumulator_value=p.INITIAL_ACCUMULATOR_VALUE,
        )

        model_manip = ModelManipulator(
            model,
            optimizer,
            loss_function,
            error_function,
            grad_mod=clip_grad_norm
        )
        train_stats, val_stats = model_manip.train(
            data,
            p.BATCH_SIZE,
            p.NUM_EPOCHS,
            dataset_val=val,
            stats_every=10,
            verbose_every=10,
            checkpoint_every=10,
            checkpoint_path=p.CHECKPOINT_PATH,
            restart=not p.CONTINUE_FROM_CHECKPOINT,
            max_steps=p.MAX_TRAINING_STEPS,
        )
        if p.CONTINUE_FROM_CHECKPOINT:
            plot_checkpoint(
                p.CHECKPOINT_PATH,
                figure_name='plot',
                show=False,
                average_over=p.AVERAGE_OVER
            )
        else:
            plot_learning_curves(
                training_values=train_stats,
                validation_values=val_stats,
                figure_name=os.path.join(p.TRAINING_PLOTS_PATH, 'plot'),
                show=False,
                average_over=p.AVERAGE_OVER
            )

    elif p.MODE == 'eval':
        model = torch.load(p.MODEL_FILE)
        produce_summary_files(
            data,
            p.DECODING_BATCH_SIZE,
            vectorizer,
            model,
            'rouge',
            beam_size=p.BEAM_SIZE,
            max_num_batch=None
        )
        # TODO: run rouge
        # run_rouge()

    elif p.MODE == 'visualize':
        model = torch.load(p.MODEL_FILE)
        batch = data[:p.DECODING_BATCH_SIZE]
        results = summarize(batch, model, beam_size=p.BEAM_SIZE)
        print_batch(batch, results[0], vectorizer)
        visualize(p.VISUALIZATION_FILE, batch, results[0], vectorizer, 0)

