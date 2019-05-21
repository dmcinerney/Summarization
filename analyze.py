import pandas as pd
import sklearn
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def lstm_chunk_generator(chunksize):
    return pd.read_csv(os.path.join('EMNLP/NSeq2SeqAttn/analysis', 'analysis_data.csv'), chunksize=chunksize)

def transformer_chunk_generator(chunksize):
    return pd.read_csv(os.path.join('EMNLP/TransformerSeq2SeqAttn3/analysis', 'analysis_data.csv'), chunksize=chunksize)

from scipy.stats import entropy
def compute_perplexity(df_chunks):
    ents = []
    for chunk_num,df in enumerate(df_chunks):
        print('CHUNKNUM', chunk_num)
        for i in range(len(df)):
            attns = np.array(eval(df['attn'].iloc[i]))
            ents.append(entropy(attns.T))
            if i % 100 == 0:
                print(i)
    ents = np.concatenate(ents)
    return np.exp(ents.mean()), np.exp(ents.mean()+ents.std())-np.exp(ents.mean())

def compute_avg_pgen(df_chunks):
    pgens = []
    for chunk_num,df in enumerate(df_chunks):
        print('CHUNKNUM', chunk_num)
        for i in range(len(df)):
            tmp_pgens = np.array(eval(df['p_gens'].iloc[i]))
            pgens.append(tmp_pgens)
            if i % 100 == 0:
                print(i)
    pgens = np.concatenate(pgens)
    return pgens.mean()

def compute_avg_copy(df_chunks):
    bools = []
    for chunk_num,df in enumerate(df_chunks):
        print('CHUNKNUM', chunk_num)
        for i in range(len(df)):
            tmp_bools = np.array(eval(df['copy_bool'].iloc[i]))
            bools.append(tmp_bools)
            if i % 100 == 0:
                print(i)
    bools = np.concatenate(bools)
    return bools.mean()

### feature_functions
def reference_length_feature(i, df, df2=None):
    return df['reference_length'].iloc[i]

def trunc_article_length_feature(i, df, df2=None):
    return min(df['article_length'].iloc[i], 399)

def article_length_feature(i, df, df2=None):
    return df['article_length'].iloc[i]

def avg_pgen(i, df):
    return np.array(eval(df['p_gens'].iloc[i])).mean()

def avg_copy(i, df):
    return np.array(eval(df['copy_bool'].iloc[i])).mean()
### end of feature functions

import scipy
def normalize(x):
    return (x-x.mean())/x.std()

def measure_correlation(df_chunks):
    score_types = ('rouge1', 'rouge2', 'rougeL')
    ys = []
    X = []
    feature_functions = [
        ('reference_length', reference_length_feature),
        ('trunc_article_length', trunc_article_length_feature),
        ('article_length', article_length_feature),
        ('avg_pgen', avg_pgen),
        ('avg_copy', avg_copy),
    ]
    for chunk_num,df in enumerate(df_chunks):
        print('CHUNKNUM', chunk_num)
        for i in range(len(df)):
            scores = [df[score_type].iloc[i] for score_type in score_types]
            ys.append(scores)
            features = [f(i, df) for n,f in feature_functions]
            X.append(features)
            if i % 100 == 0:
                print(i)
    ys = np.array(ys)
    X = np.array(X)
    feature_names = [n for n,f in feature_functions]
    correlations = []
    for i,(n,f) in enumerate(feature_functions):
        correlations.append([])
        for j,score_type in enumerate(score_types):
            correlation = scipy.stats.pearsonr(normalize(X[:,i]), normalize(ys[:,j]))
            correlations[-1].append(correlation)
    return feature_names, score_types, correlations, X, ys

def measure_difference_correlation(df1_chunks, df2_chunks):
    score_types = ('rouge1', 'rouge2', 'rougeL')
    ys = []
    X = []
    feature_functions = [
        ('reference_length', reference_length_feature),
        ('trunc_article_length', trunc_article_length_feature),
        ('article_length', article_length_feature),
    ]
    for chunk_num,(df1, df2) in enumerate(zip(df1_chunks, df2_chunks)):
        print('CHUNKNUM', chunk_num)
        for i in range(len(df1)):
            score_differences = [df1[score_type].iloc[i]-df2[score_type].iloc[i] for score_type in score_types]
            ys.append(score_differences)
            features = [f(i, df1, df2) for n,f in feature_functions]
            X.append(features)
            if i % 100 == 0:
                print(i)
    ys = np.array(ys)
    X = np.array(X)
    feature_names = [n for n,f in feature_functions]
    correlations = []
    for i,(n,f) in enumerate(feature_functions):
        correlations.append([])
        for j,score_type in enumerate(score_types):
            correlation = scipy.stats.pearsonr(normalize(X[:,i]), normalize(ys[:,j]))
            correlations[-1].append(correlation)
    return feature_names, score_types, correlations, X, ys

def print_correlations(featuren_names, score_types, correlations, X, ys, plot_name):
    for i,name in enumerate(feature_names):
        for j,score_type in enumerate(score_types):
            print(name, score_type, correlations[i][j])
            plt.plot(X[:,i], ys[:,j], '.')
            plt.savefig('analyze_generators/correlation_plots/'+plot_name+'_'+name+'_'+score_type+'.eps')
            plt.clf()

if __name__ == '__main__':
    print('compute_perplexity lstm')
    print('compute_perplexity lstm', compute_perplexity(lstm_chunk_generator(3000)))
    print('compute_avg_pgen lstm')
    print('compute_avg_pgen lstm', compute_avg_pgen(lstm_chunk_generator(3000)))
    print('compute_avg_copy lstm')
    print('compute_avg_copy lstm', compute_avg_copy(lstm_chunk_generator(3000)))
    print('measuring lstm_correlations')
    feature_names, score_types, correlations, X, ys = measure_correlation(lstm_chunk_generator(3000))
    print('lstm_correlations')
    print_correlations(feature_names, score_types, correlations, X, ys, 'lstm_correlations')

    print('compute_perplexity transformer')
    print('compute_perplexity transformer', compute_perplexity(transformer_chunk_generator(3000)))
    print('compute_avg_pgen transformer')
    print('compute_avg_pgen transformer', compute_avg_pgen(transformer_chunk_generator(3000)))
    print('compute_avg_copy transformer')
    print('compute_avg_copy transformer', compute_avg_copy(transformer_chunk_generator(3000)))
    print('measuring transformer_correlations')
    feature_names, score_types, correlations, X, ys = measure_correlation(transformer_chunk_generator(3000))
    print('transformer_correlations')
    print_correlations(feature_names, score_types, correlations, X, ys, 'transformer_correlations')

    print('measuring lstm-transformer_score_correlations')
    feature_names, score_types, correlations, X, ys = measure_difference_correlation(lstm_chunk_generator(3000), transformer_chunk_generator(3000))
    print('lstm-transformer_score_correlations')
    print_correlations(feature_names, score_types, correlations, X, ys, 'transformer-lstm_score_correlations')
