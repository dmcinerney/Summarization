from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from data import get_data, Word2VecVectorizer, TrainableVectorizer
from word_models import train_word2vec_model, save_dictionary
import os
from aspect_specific_model import AspectSummarizer
from model_helpers import aspect_summarizer_loss, aspect_summarizer_error
from pytorch_helper import ModelManipulator, TrainingTracker, plot_learning_curves, plot_checkpoint, IndicesIterator
import torch
from utils import summarize, print_batch, visualize as vis, produce_summary_files, run_rouge_1
import pdb
import parameters as p
import pickle as pkl
from model_helpers import clip_grad_norm
from submodules import TransformerTextEncoder, TransformerSummaryDecoder, LSTMTextEncoder, LSTMSummaryDecoder
import argparse

def new_model(vectorizer, aspects):
    start_index = vectorizer.word_indices['<start>']
    end_index = vectorizer.word_indices['<end>']
    return AspectSummarizer(
        vectorizer,
        start_index,
        end_index,
        aspects,
        num_hidden=p.NUM_HIDDEN,
        attn_hidden=p.ATTN_HIDDEN,
        with_coverage=p.WITH_COVERAGE,
        gamma=p.GAMMA,
        with_pointer=p.POINTER_GEN,
        encoder_base=TransformerTextEncoder if p.USE_TRANSFORMER else LSTMTextEncoder,
        decoder_base=TransformerSummaryDecoder if p.USE_TRANSFORMER else LSTMSummaryDecoder
    ).to(p.DEVICE)

def train(vectorizer, data=None, val=None):
    data = get_data(p.DATA_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE) if data is None else data
    val = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE) if val is None else val

    if p.CONTINUE_FROM_CHECKPOINT:
        # check if all of the proper files exist
        if not TrainingTracker.valid_checkpoint(p.CHECKPOINT_PATH):
            print("Cannot continue from checkpoint in \""+p.CHECKPOINT_PATH+"\" because not all of the proper files exist; restarting training.")
            p.CONTINUE_FROM_CHECKPOINT = False
            print("Saving parameters to file again.")
            p.save_params(os.path.join(p.CHECKPOINT_PATH, 'train_param_info.txt'))
        else:
            print("Loading from the last checkpoint in \""+p.CHECKPOINT_PATH+"\"; model parameters must match the saved model state.")
            if p.NEW_EPOCH:
                print("Starting from a new epoch.")
            else:
                print("Continuing from the same place in the epoch; this expects the same datafile.")

    model = new_model(vectorizer, data.dataset.aspects).train()
    if p.CONTINUE_FROM_CHECKPOINT:
        TrainingTracker.load_model_state(model, p.CHECKPOINT_PATH, map_location=p.DEVICE)

#     optimizer = torch.optim.Adam(model.parameters(), lr=p.LEARNING_RATE)
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=p.LEARNING_RATE,
        initial_accumulator_value=p.INITIAL_ACCUMULATOR_VALUE,
    )
    if p.CONTINUE_FROM_CHECKPOINT:
        TrainingTracker.load_optimizer_state(optimizer, p.CHECKPOINT_PATH, map_location=p.DEVICE)

    model_manip = ModelManipulator(
        model,
        optimizer,
        aspect_summarizer_loss,
        aspect_summarizer_error,
        grad_mod=clip_grad_norm,
        no_nan_grad=True
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


def evaluate(vectorizer, data=None):
    data = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE) if data is None else data
    model = new_model(vectorizer, data.dataset.aspects).eval()
    #with open(p.MODEL_FILE, 'rb') as modelfile:
    #    model.load_state_dict(pkl.load(modelfile))
    model.load_state_dict(torch.load(p.MODEL_FILE, map_location=p.DEVICE))
    text_path = p.TEXT_PATH
    produce_summary_files(
        data,
        p.DECODING_BATCH_SIZE,
        vectorizer,
        model,
        text_path,
        beam_size=p.BEAM_SIZE,
        max_num_batch=None
    )
    #run_rouge_2(save_to=os.path.join(p.CHECKPOINT_PATH, 'rouge_scores.txt') if p.CHECKPOINT_PATH is not None else None)
    run_rouge_1(os.path.join(text_path, 'system'), os.path.join(text_path, 'reference'), save_to=os.path.join(text_path, 'rouge_scores.txt'), verbose=True)

# very hacky
def visualize(vectorizer, data=None):
    data = get_data(p.VAL_FILE, vectorizer, with_oov=p.POINTER_GEN, aspect_file=p.ASPECT_FILE) if data is None else data
    model = new_model(vectorizer, data.dataset.aspects).eval()
    #with open(p.MODEL_FILE, 'rb') as modelfile:
    #    model.load_state_dict(pkl.load(modelfile))
    model.load_state_dict(torch.load(p.MODEL_FILE, map_location=p.DEVICE))
    vis_path = p.VIS_PATH
    for i,indices in IndicesIterator(len(data), batch_size=p.DECODING_BATCH_SIZE, shuffle=False):
        batch = data[indices]
        aspect_results = summarize(batch, model, beam_size=p.BEAM_SIZE)
        #print_batch(batch, [r[0] for r in aspect_results], vectorizer, model.aspects)
        for j in range(len(indices)):
            vis(os.path.join(vis_path, 'article%i_vis_data.json' % (i*p.DECODING_BATCH_SIZE+j)), batch, [r[0] for r in aspect_results], vectorizer, model.aspects, j, 0, pointer_gen=p.POINTER_GEN)
        print(i)

def set_params(**kwargs):
    # change parameters p based on arguments
    for k,v in kwargs.items():
        setattr(p, k.upper(), v)
    if p.CHECKPOINT_PATH is not None:
        print("Saving parameters to file.")
        p.save_params(os.path.join(p.CHECKPOINT_PATH, p.MODE+'_param_info.txt'))

def setup(**kwargs):
    set_params(**kwargs)
    if p.PRETRAINED_WORD_VECTORS:
        if not os.path.exists(p.WORD2VEC_FILE):
            train_word2vec_model(p.DATA_FILE, p.WORD2VEC_FILE, p.EMBEDDING_DIM, aspect_file=p.ASPECT_FILE)
        print('retreiving word2vec model from file '+p.WORD2VEC_FILE)
        vectorizer = Word2VecVectorizer(Word2Vec.load(p.WORD2VEC_FILE))
        if p.EMBEDDING_DIM != vectorizer.vector_size:
            raise Exception
    else:
        if not os.path.exists(p.DICTIONARY_FILE):
            save_dictionary(p.DATA_FILE, p.DICTIONARY_FILE, aspect_file=p.ASPECT_FILE)
        vectorizer = TrainableVectorizer(Dictionary.load(p.DICTIONARY_FILE), p.EMBEDDING_DIM)
    print('vocabulary length is %i' % len(vectorizer))
    return vectorizer

def main(**kwargs):
    vectorizer = setup(**kwargs)

    if p.MODE == 'train':
        print('TRAINING')
        train(vectorizer)
    elif p.MODE == 'eval':
        print('EVALUATING')
        evaluate(vectorizer)
    elif p.MODE == 'visualize':
        print('VISUALIZING')
        visualize(vectorizer)

if __name__ == '__main__':
    # set model parameters
    parser = argparse.ArgumentParser()
    for param in p.param_names:
        default = getattr(p, param)
        t_func = type(default)
        if t_func is bool:
            t_func = lambda b_string: eval(b_string)
        parser.add_argument('--'+param.lower(), default=default, type=t_func)
    args = parser.parse_args()
    kwargs = {param.lower():getattr(args, param.lower()) for param in p.param_names}
    main(**kwargs)
