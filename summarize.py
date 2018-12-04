from gensim.models import Word2Vec
from data import get_data, Vectorizer
from train_word2vec import train_word2vec_model
import os
from aspect_specific_model import AspectSummarizer
from model_helpers import aspect_summarizer_loss, aspect_summarizer_error
from pytorch_helper import ModelManipulator, TrainingTracker, plot_learning_curves, plot_checkpoint
import torch
from utils import summarize, print_batch, visualize as vis, produce_summary_files
import pdb
import parameters as p
import pickle as pkl
from model_helpers import clip_grad_norm


def train(vectorizer):
    data = get_data(p.DATA_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)
    val = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)

    if p.CONTINUE_FROM_CHECKPOINT:
        # check if all of the proper files exist
        if not TrainingTracker.valid_checkpoint(p.CHECKPOINT_PATH):
            print("Cannot continue from checkpoint because not all of the proper files exist; restarting training.")
            p.CONTINUE_FROM_CHECKPOINT = False
        else:
            print("Loading from the last checkpoint; this expects the same datafile as before.")

    if p.CONTINUE_FROM_CHECKPOINT:
        model = TrainingTracker.load_model(p.CHECKPOINT_PATH)
    else:
        start_index = vectorizer.word_indices['<start>']
        end_index = vectorizer.word_indices['<end>']
        model = AspectSummarizer(
            vectorizer,
            start_index,
            end_index,
            data.dataset.aspects,
            lstm_hidden=p.LSTM_HIDDEN,
            attn_hidden=p.ATTN_HIDDEN,
            with_coverage=p.WITH_COVERAGE,
            gamma=p.GAMMA,
            with_pointer=p.POINTER_GEN
        )

    model = model if not p.USE_CUDA else model.cuda()

#         optimizer = torch.optim.Adam(model.parameters(), lr=p.LEARNING_RATE)
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=p.LEARNING_RATE,
        initial_accumulator_value=p.INITIAL_ACCUMULATOR_VALUE,
    )
    if p.CONTINUE_FROM_CHECKPOINT:
        TrainingTracker.load_optimizer_state(optimizer, p.CHECKPOINT_PATH)

    model_manip = ModelManipulator(
        model,
        optimizer,
        aspect_summarizer_loss,
        aspect_summarizer_error,
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
        max_steps=p.MAX_TRAINING_STEPS
    )
    if p.CHECKPOINT_PATH is not None:
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
    p.save_params(os.path.join(p.CHECKPOINT_PATH, 'param_info.txt'))


def evaluate(vectorizer):
    data = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)
    model = torch.load(p.MODEL_FILE)
    produce_summary_files(
        data,
        p.DECODING_BATCH_SIZE,
        vectorizer,
        model,
        'rouge',
        beam_size=p.BEAM_SIZE,
        max_num_batch=10
    )
    # TODO: run rouge
    # run_rouge()


def visualize(vectorizer):
    data = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)
    model = torch.load(p.MODEL_FILE)
    batch = data[:p.DECODING_BATCH_SIZE]
    aspect_results = summarize(batch, model, beam_size=p.BEAM_SIZE)
    print_batch(batch, [r[0] for r in aspect_results], vectorizer, model.aspects)
    vis(p.VISUALIZATION_FILE, batch, [r[0] for r in aspect_results], vectorizer, model.aspects, 0, 0, pointer_gen=p.POINTER_GEN)


if __name__ == '__main__':
    if not os.path.exists(p.WORD2VEC_FILE):
        train_word2vec_model(p.DATA_FILE, p.WORD2VEC_FILE, p.EMBEDDING_DIM, aspect_file=p.ASPECT_FILE)
    print('retreiving word2vec model from file')
    vectorizer = Vectorizer(Word2Vec.load(p.WORD2VEC_FILE))
    print('vocabulary length is %i' % len(vectorizer))

    if p.MODE == 'train':
        print('TRAINING')
        train(vectorizer)
    elif p.MODE == 'eval':
        print('EVALUATING')
        evaluate(vectorizer)
    elif p.MODE == 'visualize':
        print('VISUALIZING')
        visualize(vectorizer)
