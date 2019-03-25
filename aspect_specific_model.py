from model import Summarizer, Encoder, Decoder, PointerGenDecoder
from submodules import LSTMTextEncoder, LSTMSummaryDecoder
from model_helpers import PointerInfo, trim_text
import parameters as p


class AspectSummarizer(Summarizer):
    def __init__(self, vectorizer, start_index, end_index, aspects, num_hidden=None, attn_hidden=None, with_coverage=False, gamma=1, with_pointer=False, encoder_base=LSTMTextEncoder, decoder_base=LSTMSummaryDecoder, decoder_parallel_base=None):
        self.aspects = aspects
        super(AspectSummarizer, self).__init__(
            vectorizer,
            start_index,
            end_index,
            num_hidden=num_hidden,
            attn_hidden=attn_hidden,
            with_coverage=with_coverage,
            gamma=gamma,
            with_pointer=with_pointer,
            encoder_base=encoder_base,
            decoder_base=decoder_base,
            decoder_parallel_base=decoder_parallel_base
        )
        self.curr_aspect = None

    def init_submodules(self):
        decoder_class = Decoder if not self.with_pointer else PointerGenDecoder
        for i,aspect in enumerate(self.aspects):
            self.__setattr__('encoder%i' % i, Encoder(self.vectorizer, self.num_hidden, encoder_base=self.encoder_base))
            self.__setattr__('decoder%i' % i, decoder_class(self.vectorizer, self.start_index, self.end_index, self.num_hidden, attn_hidden=self.attn_hidden, with_coverage=self.with_coverage, gamma=self.gamma, decoder_base=self.decoder_base, decoder_parallel_base=self.decoder_parallel_base))

    def forward(self, text, text_length, text_oov_indices=None, beam_size=1, store=None, **kwargs):
        if len(kwargs) == 0:
            for aspect in self.aspects:
                kwargs[aspect] = None
                kwargs[aspect+'_length'] = None
        if len(kwargs) != 2*len(self.aspects):
            raise Exception
        decoding = next(iter(kwargs.values())) is None
        final_return_values = {} if not decoding else []
        text, text_length = trim_text(text, text_length, p.MAX_TEXT_LENGTH)
        for i,aspect in enumerate(self.aspects):
            self.curr_aspect = i
            if store is not None:
                store[aspect] = {}
                aspect_store = store[aspect]
            else:
                aspect_store = None
            text_states, state = self.encoder(text, text_length, store=aspect_store)
            if self.with_pointer:
                self.decoder.set_pointer_info(PointerInfo(text, text_oov_indices))
            summary, summary_length = kwargs[aspect], kwargs[aspect+'_length']
            if summary is not None:
                summary, summary_length = trim_text(summary, summary_length, p.MAX_SUMMARY_LENGTH)
            return_values = self.decoder(text_states, text_length, state, summary=summary, summary_length=summary_length, beam_size=beam_size)
            if decoding:
                final_return_values.append(return_values)
            else:
                for k,v in return_values.items():
                    final_return_values[k+'_'+aspect] = v
        self.curr_aspect = None
        return final_return_values

    @property
    def encoder(self):
        if self.curr_aspect is None:
            raise Exception
        return self.__getattr__('encoder%i' % self.curr_aspect)

    @property
    def decoder(self):
        if self.curr_aspect is None:
            raise Exception
        return self.__getattr__('decoder%i' % self.curr_aspect)
