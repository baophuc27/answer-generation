import numpy as np
import re

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences


class Vocab(object):
    def __init__(self,sentences):
        self.token_to_ix = {}
        self.ix_to_token = {}
        self._count = 0

        for token in [PAD_TOKEN,UNKNOWN_TOKEN,START_DECODING,STOP_DECODING]:
            self.token_to_ix[token] = self._count
            self.ix_to_token[self._count] = token
            self._count +=1
        
        for item in sentences:
            sent = list(item.values())[0]
            tokens = re.sub(
                    r"([.,'!?\"()*#:;])",'',
                    sent.lower()
                    ).replace('-', ' ').replace('/', ' ').split()
            for token in tokens:
                if token not in self.token_to_ix:
                    self.token_to_ix[token] = self._count
                    self.ix_to_token[self._count] = token
                    self._count += 1
        print("=== Total vocab size: ",str(self._count))

    def token_to_ix(token):
        if token in self.token_to_ix:
            return self.token_to_ix[token]
        else:
            return self.token_to_ix[UNKNOWN_TOKEN]
    
    def ix_to_token(ix):
        if ix not in self.ix_to_token:
            raise ValueError("Invalid index")
        return self.ix_to_token[ix]
    
    def vocab_size(self):
        return self._count + 1
        


