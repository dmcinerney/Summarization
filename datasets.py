from pytorch_helper import VariableLength, OneAtATimeDataset, pad_and_concat, dicts_into_batch
import spacy
from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))
import pdb

class SummarizationDataset:
    def __init__(self, df, aspect_file=None):
        self.df = df
        if aspect_file is None:
            # if aspect_file is not present, default to one aspect with name 'summary'
            self.aspects = ['summary']
        else:
            with open(aspect_file, 'r') as aspectfile:
                self.aspects = eval(aspectfile.read())

    def read(self, i):
        text = self.prepare_text(self.df.text[i])
        aspect_summaries = [self.prepare_text(self.df[aspect_name][i]) for aspect_name in self.aspects]
        return (text, *aspect_summaries)

    def prepare_text(self, text):
        return ['<start>'] + text + ['<end>']

    def text_iterator(self):
        return self.TextIterator(self)

    def __len__(self):
        return len(self.df)

    class TextIterator:
        def __init__(self, dataset):
            self.dataset = dataset

        def __iter__(self):
            for i in range(len(self.dataset)):
                texts = self.dataset.read(i)
                for text in texts:
                    yield text

class PreprocessedSummarizationDataset(VariableLength):
    def __init__(self, dataset, vectorizer, with_oov=False):
        self.dataset = dataset
        self.vectorizer = vectorizer
        self.with_oov = with_oov
        
    def get_raw_inputs(self, i):
        return self.dataset.read(i), None
    
    def prepare_inputs(self, v_args, nv_args, lengths):
        text, summaries = v_args[0], v_args[1:]
        text_length_max, summary_length_maxes = lengths[0], lengths[1:]
        text, text_length, text_oov_indices = self.vectorizer.get_text_indices(text, text_length_max)
        return_dict = dict(text=text, text_length=text_length)
        if self.with_oov:
            return_dict['text_oov_indices'] = text_oov_indices
        for i,summary in enumerate(summaries):
            summary, summary_length, _ = self.vectorizer.get_text_indices(summary, summary_length_maxes[i], oov_indices=text_oov_indices)
            return_dict[self.dataset.aspects[i]] = summary
            return_dict[self.dataset.aspects[i]+'_length'] = summary_length
        return return_dict
    
    def __len__(self):
        return len(self.dataset)

class HierarchicalDataset(OneAtATimeDataset):
    def __init__(self, dataset, vectorizer, with_oov=False):
        self.dataset = dataset
        self.vectorizer = vectorizer
        self.with_oov = with_oov
        
    def get_one_item(self, i):
        text, summary = self.dataset.read(i)
        sentences = [[text[0]]]+[str(sent).split() for sent in nlp(' '.join(text[1:])).sents]
        oov_indices = {}
        text_indices = []
        lengths = []
        for sentence in sentences:
            text, text_length, oov_indices = self.vectorizer.get_text_indices(sentence, len(sentence), oov_indices=oov_indices)
            text_indices.append(text)
            lengths.append(text_length)
        summary_indices, summary_length, _ = self.vectorizer.get_text_indices(summary, len(summary), oov_indices=oov_indices)
        return_dict = dict(text=pad_and_concat(text_indices), text_length=pad_and_concat(lengths, static=True), summary=summary_indices, summary_length=summary_length)
        if self.with_oov:
            return_dict['text_oov_indices'] = oov_indices
        return return_dict

    def get_multiple_items(self, index_generator):
        list_of_dicts = [self[i] for i in index_generator]
        return dicts_into_batch(list_of_dicts, variable_keys=['text', 'summary', 'text_length'])
    
    def __len__(self):
        return len(self.dataset)
