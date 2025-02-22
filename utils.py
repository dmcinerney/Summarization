from gensim.models import Word2Vec
import os.path
import spacy, re
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from subprocess import call
from pytorch_helper import IndicesIterator
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
import pdb

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = [t.text.lower() for t in nlp(text)]
    text = ['qqq' if any(char.isdigit() for char in word) else word for word in text]
    return text

def produce_attention_visualization_file(filename, text, reference_summary, decoded_summary, attentions, p_gens):
    data = {'article_lst': text,
            'decoded_lst': decoded_summary,
            'abstract_str': reference_summary,
            'attn_dists': attentions,
            'p_gens': p_gens}

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def summarize(batch, model, beam_size=1, store=None):
    device = list(model.parameters())[0].device
    if model.with_pointer:
        aspect_results = model(batch['text'].to(device), batch['text_length'].to(device), batch['text_oov_indices'], beam_size=beam_size, store=store)
    else:
        aspect_results = model(batch['text'].to(device), batch['text_length'].to(device), beam_size=beam_size, store=store)
    return aspect_results

def get_text_triplets(batch, aspect_summaries, vectorizer, aspects):
    text_triplets = []
    for i in range(len(aspect_summaries[0][0])):
        text_indices, text_length = batch['text'][i].numpy(), batch['text_length'][i].numpy()
        oov_words = {v:k for k,v in batch['text_oov_indices'][i].items()} if 'text_oov_indices' in batch.keys() else None
        text = vectorizer.get_index_words(text_indices[:text_length], oov_words=oov_words)

        reference_summaries, decoded_summaries = [], []
        for j,aspect in enumerate(aspects):
            r_summary_indices, r_summary_length = batch[aspect][i].numpy(), batch[aspect+'_length'][i].numpy()
            summary_indices, summary_length = aspect_summaries[j][0][i], aspect_summaries[j][1][i]
            reference_summary = vectorizer.get_index_words(r_summary_indices[:r_summary_length], oov_words=oov_words)
            decoded_summary = vectorizer.get_index_words(summary_indices[:summary_length], oov_words=oov_words)
            reference_summaries.append(reference_summary)
            decoded_summaries.append(decoded_summary)

        text_triplets.append((text,reference_summaries,decoded_summaries))
    return text_triplets

def postprocess(text_tokens):
    return [token for token in text_tokens if token[0] != '<' or token[-1] != '>']

def rouge_preprocess(text_tokens):
    text_tokens = postprocess(text_tokens)
    return " ".join(text_tokens).replace(" . ", " .\n").replace(" ! ", " !\n").replace(" ? ", " ?\n")

def produce_batch_summary_files(batch, vectorizer, model, path, beam_size=1, start_index=0):
    aspect_results = summarize(batch, model, beam_size=beam_size)
    text_triplets = get_text_triplets(batch, [r[0] for r in aspect_results], vectorizer, model.aspects)
    i = start_index
    for text,reference_summaries,decoded_summaries in text_triplets:
        with open(os.path.join(path,"articles/article"+str(i)+"_text.txt"), "w") as textfile:
            textfile.write(rouge_preprocess(text))
        for j,aspect in enumerate(model.aspects):
            reference = reference_summaries[j]
            decoded = decoded_summaries[j]
            with open(os.path.join(path, "reference/article"+str(i)+"_reference_"+aspect+".txt"), "w") as referencefile:
                referencefile.write(rouge_preprocess(reference))
            with open(os.path.join(path, "system/article"+str(i)+"_system_"+aspect+".txt"), "w") as decodedfile:
                decodedfile.write(rouge_preprocess(decoded))
        i += 1
    return i

def produce_summary_files(dataset, batch_size, vectorizer, model, path, beam_size=1, max_num_batch=None):
    start_index = 0
    for i,indices in IndicesIterator(len(dataset), batch_size=batch_size, shuffle=False):
        batch = dataset[indices]
        start_index = produce_batch_summary_files(batch, vectorizer, model, path, beam_size=beam_size, start_index=start_index)
        print(i)
        if max_num_batch is not None and (i+1) >= max_num_batch:
            break

def run_rouge_1(system_path, reference_path, save_to=None):
    from pyrouge import Rouge155
    r = Rouge155()
    r.system_dir = system_path
    r.model_dir = reference_path
    r.system_filename_pattern = 'article(\d+)_system_summary.txt'
    r.model_filename_pattern = 'article#ID#_reference_summary.txt'

    output = r.convert_and_evaluate()
    print(output)
    output_dict = r.output_to_dict(output)
    if save_to is not None:
        with open(save_to, 'w') as file:
            file.write(output)
            file.write(str(output_dict))

def run_rouge_2(save_to=None):
    call(['java', '-Drouge.prop=rouge/ROUGE-2/rouge.properties', '-jar', 'rouge/ROUGE-2/rouge2-1.2.jar'])
    df = pd.read_csv("rouge/ROUGE-2/results.csv")
    rouge1 = df[df['ROUGE-Type'] == 'ROUGE-1+StopWordRemoval']['Avg_F-Score'].mean()*100
    rouge2 = df[df['ROUGE-Type'] == 'ROUGE-2+StopWordRemoval']['Avg_F-Score'].mean()*100
    rougel = df[df['ROUGE-Type'] == 'ROUGE-L+StopWordRemoval']['Avg_F-Score'].mean()*100
    rouge_readout = 'ROUGE-1: %f, ROUGE-2: %f, ROUGE-L: %f' % (rouge1, rouge2, rougel)
    print(rouge_readout)
    if save_to is not None:
        with open(save_to, 'w') as file:
            file.write(rouge_readout+'\n')

def print_batch(batch, aspect_summaries, vectorizer, aspects):
    triplets = get_text_triplets(batch, aspect_summaries, vectorizer, aspects)
    for i,(text, reference_summaries, decoded_summaries) in enumerate(triplets):
        print("text", text)
        for j,aspect in enumerate(aspects):
            reference_summary = reference_summaries[j]
            decoded_summary = decoded_summaries[j]
            print(aspect+" reference summary", reference_summary)
            print(aspect+" decoded summary", decoded_summary)
            loss = aspect_summaries[j][2][i]
            print(loss)
        print('')

def visualize(filename, batch, aspect_summaries, vectorizer, aspects, i, j, pointer_gen=False):
    triplets = get_text_triplets(batch, aspect_summaries, vectorizer, aspects)
    text, reference_summaries, decoded_summaries = triplets[i]
    reference_summary, decoded_summary = reference_summaries[j], decoded_summaries[j]
    attentions = [[float(f) for f in vector[1:]] for vector in aspect_summaries[j][4][i]]
    p_gens = [float(f) for f in aspect_summaries[j][5][i]] if pointer_gen else [1. for i in range(len(attentions))]

    produce_attention_visualization_file(filename, text[1:-1], " ".join(reference_summary[1:-1]), decoded_summary[1:], attentions, p_gens)

def inspect_model_for_nans(model):
    for n,p in model.named_parameters():
        if (p != p).any():
            if (p != p).all():
                outstr = "all"
            else:
                outstr = "some"
        else:
            outstr = "none"
        print("%s: %s are NaN" % (n, outstr))

def extract_checkpoint_rouge(checkpoints):
    scores = []
    timesteps = []
    for checkpoint in checkpoints:
        with open(os.path.join(checkpoint, 'rouge_scores.txt'), 'r') as f:
            for line in f:
                pass
        rouge_dict = eval(line)
        with open(os.path.join(checkpoint, 'iternum.txt'), 'r') as f:
            _, timestep = eval(f.read())
        scores.append(rouge_dict)
        timesteps.append(timestep)
    return timesteps, scores

def plot_stacked_bar(data, x_ticks=None, stack_labels=None, y_label=None, title=None, show_nums=None, y_lim=None, file=None, figsize=None):
    ind = np.arange(len(data[0][0]))    # the x locations for the groups
    width = 0.40       # the width of the bars: can also be len(x) sequence
    if figsize is not None:
        plt.figure(figsize=figsize)
    ps = [plt.bar(
        ind,
        mean,
        width,
        bottom=(data[i-1][0] if i > 0 else None),
        yerr=error
    ) for i,(mean,error) in enumerate(data)]

    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    if x_ticks is not None:
        plt.xticks(ind, x_ticks)
    if stack_labels is not None:
        plt.legend(tuple(p[0] for p in ps), stack_labels)
    
    for i,bar in enumerate(ps):
        for j,patch in enumerate(bar):
            if show_nums is None or show_nums[i,j]:
            # get_width pulls left or right; get_y pushes up or down
                plt.text(patch.get_x(), sum(p[j].get_height() for p in ps[:i+1])+.005, \
                        str(round(sum(mean[j] for mean,_ in data[:i+1]), 4)), fontsize=12)
    if y_lim is not None:
        plt.ylim(y_lim)
    if file is not None:
        plt.savefig(file)