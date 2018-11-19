from gensim.models import Word2Vec
import os.path
import spacy, re
import torch
import numpy as np
import json
from pyrouge import Rouge155
import pandas as pd
from subprocess import call
from pytorch_helper import IndicesIterator
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

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
        
def summarize(batch, model, beam_size=1):
    device = list(model.parameters())[0].device
    if model.with_pointer:
        results = model(batch['text'].to(device), batch['text_length'].to(device), batch['text_oov_indices'], beam_size=beam_size)
    else:
        results = model(batch['text'].to(device), batch['text_length'].to(device), beam_size=beam_size)
    return results
    
def get_text_triplets(batch, summary_info, vectorizer, pointer_gen=False):
    text_triplets = []
    for i in range(len(summary_info[0])):
        summary_indices, summary_length = summary_info[0][i], summary_info[1][i]
        r_summary_indices, r_summary_length = batch['summary'][i].numpy(), batch['summary_length'][i].numpy()
        text_indices, text_length = batch['text'][i].numpy(), batch['text_length'][i].numpy()
        oov_words = {v:k for k,v in batch['text_oov_indices'][i].items()} if pointer_gen is True else None

        text = vectorizer.get_index_words(text_indices[:text_length], oov_words=oov_words)
        reference_summary = vectorizer.get_index_words(r_summary_indices[:r_summary_length], oov_words=oov_words)
        decoded_summary = vectorizer.get_index_words(summary_indices[:summary_length], oov_words=oov_words)
        
        text_triplets.append((text,reference_summary,decoded_summary))
    return text_triplets

def rouge_preprocess(text_tokens):
    return " ".join(text_tokens).replace(" . ", " .\n").replace(" ! ", " !\n").replace(" ? ", " ?\n")

def produce_batch_summary_files(batch, vectorizer, model, path, beam_size=1, start_index=0):
    results = summarize(batch, model, beam_size=beam_size)
    text_triplets = get_text_triplets(batch, results[0], vectorizer, pointer_gen=model.with_pointer)
    i = start_index
    for text,reference,decoded in text_triplets:
        with open(os.path.join(path,"articles/article"+str(i)+"_text.txt"), "w") as textfile:
            textfile.write(rouge_preprocess(text[1:-1]))
        with open(os.path.join(path,"reference/article"+str(i)+"_reference.txt"), "w") as referencefile:
            referencefile.write(rouge_preprocess(reference[1:-1]))
        with open(os.path.join(path,"system/article"+str(i)+"_system.txt"), "w") as decodedfile:
            decodedfile.write(rouge_preprocess(decoded[1:-1]))
        i += 1
    return i
    
def produce_summary_files(dataset, batch_size, vectorizer, model, path, beam_size=1, max_num_batch=None):
    start_index = 0
    for i,indices in IndicesIterator(len(dataset), batch_size=batch_size, shuffle=True):
        batch = dataset[indices]
        start_index = produce_batch_summary_files(batch, vectorizer, model, path, beam_size=beam_size, start_index=start_index)
        print(i)
        if max_num_batch is not None and (i+1) >= max_num_batch:
            break

def run_rouge():
#     call(["-Drouge.prop=rouge/ROUGE-2/rouge.properties"])
#     call(["java", "-jar", "rouge/ROUGE-2/rouge2-1.2.jar"])
    df = pd.read_csv("rouge/ROUGE-2/results.csv")
    rouge1 = df[df['ROUGE-Type'] == 'ROUGE-1+StopWordRemoval']['Avg_F-Score'].mean()*100
    rouge2 = df[df['ROUGE-Type'] == 'ROUGE-2+StopWordRemoval']['Avg_F-Score'].mean()*100
    rougel = df[df['ROUGE-Type'] == 'ROUGE-L+StopWordRemoval']['Avg_F-Score'].mean()*100
    print(rouge1, rouge2, rougel)
    return df

def print_batch(batch, summary_info, vectorizer):
    triplets = get_text_triplets(batch, summary_info, vectorizer, pointer_gen=True)
    for i,(text, reference_summary, decoded_summary) in enumerate(triplets):
        loss = summary_info[2][i]
        print("text", text)
        print("reference summary", reference_summary)
        print("decoded summary", decoded_summary)
        print(loss)
        
def visualize(filename, batch, summary_info, vectorizer, i):
    triplets = get_text_triplets(batch, summary_info, vectorizer)
    text, reference_summary, decoded_summary = triplets[i]
    attentions, p_gens = [[float(f) for f in vector[1:-1]] for vector in summary_info[4][i][:-1]], [float(f) for f in summary_info[5][i][:-1]]
    produce_attention_visualization_file(filename, text[1:-1], " ".join(reference_summary[1:-1]), decoded_summary[1:-1], attentions, p_gens)
