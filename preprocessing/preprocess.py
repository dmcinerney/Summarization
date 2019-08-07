import struct
from newsroom import jsonl
import os
from utils import preprocess_text
import parameters as p
import pandas as pd
import pdb
import pickle as pkl
from tensorflow.core.example import example_pb2

def trim_and_transform(example_generator, new_filename, transformation, constraint):
    oldcount, newcount = 0, 0
    if os.path.isfile(new_filename):
        os.remove(new_filename)
    with jsonl.open(new_filename, gzip=True) as newfile:
        for line in example_generator:
            oldcount += 1
            line = transformation(line)
            if constraint(line):
                newcount += 1
                newfile.appendline(line)
            if oldcount % 1000 == 0:
                print(oldcount)
    print('# of old lines: %i, # of new lines: %i' % (oldcount, newcount))

def newsroom_constraint(line):
    return line['text'] is not None and line['summary'] is not None

def newsroom_preprocess(line):
    text = preprocess_text(line['text']) if line['text'] is not None else None
    summary = preprocess_text(line['summary']) if line['summary'] is not None else None
    return dict(text=text, summary=summary)

# # TODO: clean this up
# def cnn_preprocess(example_str):
#     abstract = []
#     article = []
#     in_abstract = False
#     in_article = False
#     example_str = example_str.decode('utf-8', 'replace')
#     example_str = example_str.replace('<s>', ' ')
#     example_str = example_str.replace('</s>', ' . ')
#     prev_c = None
#     for c in example_str.split():
#         if c == '.' and prev_c == c:
#             continue
#         if 'abstract' in c and c != 'abstract':
#             in_abstract = True
#             in_article = False
#             continue
#         if 'article' in c and c != 'article':
#             in_abstract = False
#             in_article = True
#             continue
#         c = c.replace('</s>', '.')
#         if '<s>' in c: continue
#         if '�' in c: continue
#         if in_abstract:
#             abstract.append(c)
#         if in_article:
#             article.append(c)
#         prev_c = c
#     pdb.set_trace()
#     return dict(text=article, summary=abstract)
def cnn_preprocess(example_str):
    # convert to tensorflow example e
    e = example_pb2.Example.FromString(example_str)
    # extract text and summary
    try:
        # the article text was saved under the key 'article' in the data files
        article_text = e.features.feature['article'].bytes_list.value[0].decode().split(' ')
        # the abstract text was saved under the key 'abstract' in the data files
        abstract_text = e.features.feature['abstract'].bytes_list.value[0].decode().split(' ')
    except ValueError:
        article_text = abstract_text = None
    return dict(text=article_text, summary=abstract_text)

def cnn_constraint(line):
    return line['text'] is not None and line['summary'] is not None

def pico_preprocess(line):
    line = dict(text=line.abstract, P=line.population, I=line.intervention, O=line.outcome)
    if pico_constraint(line):
        return {k:preprocess_text(v) for k,v in line.items()}
    else:
        return line

def pico_constraint(line):
	return line['text'] == line['text'] and \
	       line['P'] == line['P'] and \
	       line['I'] == line['I'] and \
	       line['O'] == line['O']

def preprocess_newsroom_datafile(filename, new_filename):
    with jsonl.open(filename, gzip=True) as oldfile:
        trim_and_transform(oldfile, new_filename, newsroom_preprocess, newsroom_constraint)

def preprocess_cnn_datafile(filename, new_filename):
    def cnn_dataset_generator():
        with open(filename, "rb") as file:
            while True:
                len_bytes = file.read(8)
                if not len_bytes: break # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, file.read(str_len))[0]
                yield example_str
    trim_and_transform(cnn_dataset_generator(), new_filename, cnn_preprocess, cnn_constraint)

def preprocess_pico_dataset(filename, new_filename_train, new_filename_dev, new_filename_test, aspect_file):
    df = pd.read_csv(filename)
    def train_generator():
        for i,row in df[:30000].iterrows():
            yield row
    def dev_generator():
        for i,row in df[30000:40000].iterrows():
            yield row
    def test_generator():
        for i,row in df[40000:].iterrows():
            yield row
    trim_and_transform(train_generator(), new_filename_train, pico_preprocess, pico_constraint)
    trim_and_transform(dev_generator(), new_filename_dev, pico_preprocess, pico_constraint)
    trim_and_transform(test_generator(), new_filename_test, pico_preprocess, pico_constraint)
    with open(aspect_file, 'w') as aspectfile:
        aspectfile.write(str(['P','I','O']))

if __name__ == '__main__':
    # for newsroom dataset
    # filename = 'data/newsroom_dataset/train.data'
    # new_filename = 'data/newsroom_dataset/train_processed.data'
    # preprocess_newsroom_datafile(filename, new_filename)

    # for cnn dataset
    filename = 'data/cnn_dataset/test.bin'
    new_filename = 'data/cnn_dataset/test_processed.data'
    preprocess_cnn_datafile(filename, new_filename)

    # for pico dataset
#     aspect_file = '/Volumes/JEREDUSB/aspects.txt'
    # filename = '/Volumes/JEREDUSB/pico_cdsr.csv'
    # new_filename_train = '/Volumes/JEREDUSB/train_processed.data'
    # new_filename_dev = '/Volumes/JEREDUSB/dev_processed.data'
    # new_filename_test = '/Volumes/JEREDUSB/test_processed.data'
    # preprocess_pico_dataset(filename, new_filename_train, new_filename_dev, new_filename_test, aspect_file)
#     with open(aspect_file, 'w') as aspectfile:
#         aspectfile.write(str(['P','I','O']))
