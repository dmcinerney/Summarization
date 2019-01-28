from gensim.models import Word2Vec
from data import get_data, Word2VecVectorizer
from train_word2vec import train_word2vec_model
import os
from aspect_specific_model import AspectSummarizer
from model_helpers import aspect_summarizer_loss, aspect_summarizer_error
from pytorch_helper import ModelManipulator, TrainingTracker, plot_learning_curves, plot_checkpoint
import torch
from utils import summarize, print_batch, visualize as vis, produce_summary_files, run_rouge
import pdb
import parameters as p
import pickle as pkl
from model_helpers import clip_grad_norm
from submodules import TransformerTextEncoder, TransformerSummaryDecoder, LSTMTextEncoder, LSTMSummaryDecoder

def new_model(vectorizer, aspects):
    start_index = vectorizer.word_indices['<start>']
    end_index = vectorizer.word_indices['<end>']
    return AspectSummarizer(
        vectorizer,
        start_index,
        end_index,
        aspects,
        lstm_hidden=p.LSTM_HIDDEN,
        attn_hidden=p.ATTN_HIDDEN,
        with_coverage=p.WITH_COVERAGE,
        gamma=p.GAMMA,
        with_pointer=p.POINTER_GEN,
        encoder_base=TransformerTextEncoder if p.USE_TRANSFORMER else LSTMTextEncoder,
        decoder_base=TransformerSummaryDecoder if p.USE_TRANSFORMER else LSTMSummaryDecoder
    )

def train(vectorizer):
    data = get_data(p.DATA_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)
    val = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)

    if p.CONTINUE_FROM_CHECKPOINT:
        # check if all of the proper files exist
        if not TrainingTracker.valid_checkpoint(p.CHECKPOINT_PATH):
            print("Cannot continue from checkpoint because not all of the proper files exist; restarting training.")
            p.CONTINUE_FROM_CHECKPOINT = False
            print("Saving parameters to file again.")
            p.save_params(os.path.join(p.CHECKPOINT_PATH, 'param_info.txt'))
        else:
            print("Loading from the last checkpoint; model parameters must match the saved model state.")
            if p.NEW_EPOCH:
                print("Starting from a new epoch.")
            else:
                print("Continuing from the same place in the epoch; this expects the same datafile.")

    model = new_model(vectorizer, data.dataset.aspects).train()
    if p.CONTINUE_FROM_CHECKPOINT:
        TrainingTracker.load_model_state_(model, p.CHECKPOINT_PATH)
        pdb.set_trace()

    model = model if not p.USE_CUDA else model.cuda()

#     optimizer = torch.optim.Adam(model.parameters(), lr=p.LEARNING_RATE)
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
    with torch.autograd.set_detect_anomaly(p.DETECT_ANOMALY):
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
            new_epoch=p.NEW_EPOCH,
            max_steps=p.MAX_TRAINING_STEPS,
            save_whole_model=False
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


def evaluate(vectorizer):
    data = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)
    model = new_model(vectorizer, data.dataset.aspects).eval()
    with open(p.MODEL_FILE, 'rb') as modelfile:
        model.load_state_dict(pkl.load(modelfile))
    produce_summary_files(
        data,
        p.DECODING_BATCH_SIZE,
        vectorizer,
        model,
        'rouge',
        beam_size=p.BEAM_SIZE,
        max_num_batch=None
    )
    run_rouge(save_to=os.path.join(p.CHECKPOINT_PATH, 'rouge_scores.txt') if p.CHECKPOINT_PATH is not None else None)


def visualize(vectorizer):
    data = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE)
    model = new_model(vectorizer, data.dataset.aspects).eval()
    with open(p.MODEL_FILE, 'rb') as modelfile:
        model.load_state_dict(pkl.load(modelfile))
    batch = data[:p.DECODING_BATCH_SIZE]
    aspect_results = summarize(batch, model, beam_size=p.BEAM_SIZE)
    print_batch(batch, [r[0] for r in aspect_results], vectorizer, model.aspects)
    vis(p.VISUALIZATION_FILE, batch, [r[0] for r in aspect_results], vectorizer, model.aspects, 0, 0, pointer_gen=p.POINTER_GEN)


if __name__ == '__main__':
    print("Saving parameters to file.")
    p.save_params(os.path.join(p.CHECKPOINT_PATH, 'param_info.txt'))
    if not os.path.exists(p.WORD2VEC_FILE):
        train_word2vec_model(p.DATA_FILE, p.WORD2VEC_FILE, p.EMBEDDING_DIM, aspect_file=p.ASPECT_FILE)
    print('retreiving word2vec model from file')
    vectorizer = Word2VecVectorizer(Word2Vec.load(p.WORD2VEC_FILE))
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
