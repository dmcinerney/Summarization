import pandas as pd
import numpy as np
import os
from shutil import copyfile
import pdb
import nltk
from utils import run_rouge_1
import argparse
import spacy
import json
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

###### Start of Feature Functions
def length(a, r, s):
    def num_words(text):
        return sum(len(sent.split()) for sent in text)
    pdb.set_trace
    return {'article_length':num_words(a), 'reference_length':num_words(r), 'system_length':num_words(s)}

def depth(a, r, s):
    def dependency_depth(text):
        return 0
    return {'article_depth':dependency_depth(a), 'reference_depth':dependency_depth(r), 'system_depth':dependency_depth(s)}

def num_sentences(a, r, s):
    num_article_sentences = len(list(nlp(" ".join(a[0].split(' ')[1:-1])).sents))
    return {'num_article_sentences':num_article_sentences, 'num_reference_sentences':len(r), 'num_system_sentences':len(s),}

def copy_boolean(vis_data):
    vis_data['article_lst'] = vis_data['article_lst']+['<end>']
    article_idxs = np.array(vis_data['attn_dists']).argmax(1)[:len(vis_data['decoded_lst'])]
    bools = [vis_data['article_lst'][article_idx]==vis_data['decoded_lst'][decoded_idx] for decoded_idx,article_idx in enumerate(article_idxs)]
    return {'copy_bool':bools}

def attention(vis_data):
    return {'attn':vis_data['attn_dists'][:len(vis_data['decoded_lst'])]}

def p_gens(vis_data):
    return {'p_gens':vis_data['p_gens'][:len(vis_data['decoded_lst'])]}
###### End of Feature Functions

def read_text(file):
    with open(file, 'r') as f:
        return [line.strip() for line in f]

def read_vis(file):
    with open(file, 'r') as f:
        return json.load(f)

def analyze(article_file, reference_summary_file, system_summary_file, vis_file):
    article = read_text(article_file)
    reference_summary = read_text(reference_summary_file)
    system_summary = read_text(system_summary_file)
    vis_data = read_vis(vis_file)
    feature_funcs = [length, depth, num_sentences]
    vis_feature_funcs = [copy_boolean, attention, p_gens]
    features = {}
    for func in feature_funcs:
        features.update(func(article, reference_summary, system_summary))
    for func in vis_feature_funcs:
        features.update(func(vis_data))
    return features

def append_to_file(df, file):
    header = not os.path.isfile(file)
    with open(file, 'a') as f:
        df.to_csv(f, header=header)

def process_summaries(articles_path, reference_path, system_path, directory):
    os.mkdir(directory)
    tmp_reference = os.path.join(directory, 'system')
    tmp_system = os.path.join(directory, 'reference')
    os.mkdir(tmp_reference)
    os.mkdir(tmp_system)
    rows = []
    for i,(a,rs,ss) in list(enumerate(zip(sorted(os.listdir(articles_path)), sorted(os.listdir(reference_path)), sorted(os.listdir(system_path))))):
        print(i, a, rs, ss)
        copyfile(os.path.join(reference_path, rs), os.path.join(tmp_reference, rs))
        copyfile(os.path.join(system_path, ss), os.path.join(tmp_system, ss))
        output = run_rouge_1(tmp_system, tmp_reference, verbose=False)
        rouge_scores = {'rouge1':output['rouge_1_f_score'],'rouge2':output['rouge_2_f_score'],'rougeL':output['rouge_l_f_score']}
        vis_file = os.path.join(system_path+'_vis', 'article%i_vis_data.json' % i)
        features = analyze(os.path.join(articles_path, a), os.path.join(reference_path, rs), os.path.join(system_path, ss), vis_file)
        row = {**rouge_scores, **features}
        rows.append(row)
        # pdb.set_trace()
        os.remove(os.path.join(tmp_reference, rs))
        os.remove(os.path.join(tmp_system, ss))
        if (i+1) % 100 == 0:
            print('UPDATING FILE')
            append_to_file(pd.DataFrame(rows), os.path.join(directory, 'analysis_data.csv'))
            rows = []
    append_to_file(pd.DataFrame(rows), os.path.join(directory, 'analysis_data.csv'))
    os.rmdir(tmp_reference)
    os.rmdir(tmp_system)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('system_path', type=str)
    parser.add_argument('temp_analysis_dir', type=str)
    args = parser.parse_args()

    process_summaries(
        articles_path='articles',
        reference_path='reference',
        system_path=args.system_path,
        directory=args.temp_analysis_dir
    )