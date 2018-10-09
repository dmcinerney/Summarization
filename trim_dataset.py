from newsroom import jsonl
import os
from os import path
from utils import preprocess_text

def line_length_constraint(line):
    return line['text'] is not None and line['summary'] is not None and len(line['text']) <= 500

def preprocess_text_transformation(line):
    text = preprocess_text(line['text']) if line['text'] is not None else None
    summary = preprocess_text(line['summary']) if line['summary'] is not None else None
    return dict(text=text, summary=summary)

def trim(old_filename, new_filename, transformation, constraint):
    oldcount, newcount = 0, 0
    if path.isfile(new_filename):
        os.remove(new_filename)
    with jsonl.open(new_filename, gzip=True) as newfile:
        with jsonl.open(old_filename, gzip=True) as oldfile:
            for line in oldfile:
                oldcount += 1
                line = transformation(line)
                if constraint(line):
                    newcount += 1
                    newfile.appendline(line)
                if oldcount % 1000 == 0:
                    print(oldcount)
    print('# of old lines: %i, # of new lines: %i' % (oldcount, newcount))

if __name__ == '__main__':
    trim('data/dev.data', 'data/dev_trimmed.data', preprocess_text_transformation, line_length_constraint)
