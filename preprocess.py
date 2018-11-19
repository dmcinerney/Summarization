import struct
from newsroom import jsonl
import os
from utils import preprocess_text
import parameters as p

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
    return line['text'] is not None and line['summary'] is not None and len(line['text']) <= 500

def newsroom_preprocess(line):
    text = preprocess_text(line['text']) if line['text'] is not None else None
    summary = preprocess_text(line['summary']) if line['summary'] is not None else None
    return dict(text=text, summary=summary)

# TODO: clean this up
def cnn_preprocess(example_str):
    abstract = []
    article = []
    in_abstract = False
    in_article = False
    example_str = example_str.decode('utf-8', 'replace')
    example_str = example_str.replace('<s>', ' ')
    example_str = example_str.replace('</s>', ' . ')
    prev_c = None
    for c in example_str.split():
        if c == '.' and prev_c == c:
            continue
        if 'abstract' in c and c != 'abstract':
            in_abstract = True
            in_article = False
            continue
        if 'article' in c and c != 'article':
            in_abstract = False
            in_article = True
            continue
        c = c.replace('</s>', '.')
        if '<s>' in c: continue
        if '�' in c: continue
        if in_abstract:
            abstract.append(c)
        if in_article:
            article.append(c)
        prev_c = c
    return dict(text=article, summary=abstract)

def cnn_constraint(line):
    return len(line['text']) > 0 and len(line['summary']) > 0 and len(line['text']) <= 500

if __name__ == '__main__':
    # for newsroom dataset
#     filename = 'data/newsroom_dataset/train.data'
#     new_filename = 'data/newsroom_dataset/train_processed.data'
#     with jsonl.open(filename, gzip=True) as oldfile:
#         trim_and_transform(oldfile, new_filename, newsroom_preprocess, newsroom_constraint)

    # for cnn dataset
    filename = 'data/cnn_dataset/val.bin'
    new_filename = 'data/cnn_dataset/val_processed.data'
    def cnn_dataset_generator(filename):
        with open(filename, "rb") as trainfile:
            while True:
                len_bytes = trainfile.read(8)
                if not len_bytes: break # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, trainfile.read(str_len))[0]
                yield example_str

    trim_and_transform(cnn_dataset_generator(filename), new_filename, cnn_preprocess, cnn_constraint)
