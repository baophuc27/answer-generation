import numpy as np
import re
from collections import Counter


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

        total_tokens = []   
        
        for item in sentences:
            sent = list(item.values())[0]
            tokens = re.sub(
                    r"([.,'!\"()*#:;])",'',
                    sent.lower()
                    ).replace('-', ' ').replace('/', ' ').split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        # Only token with at least 10 occurrences can be in vocab
        filter_tokens =  list(filter(lambda x : counter[x] > 10 , counter))
        for token in [PAD_TOKEN,UNKNOWN_TOKEN,STOP_DECODING,START_DECODING]:
            filter_tokens.insert(0,token)

        for ix,token in enumerate(filter_tokens):
            self.token_to_ix[token] = ix
            self.ix_to_token[ix] = token
        self._count = len(self.token_to_ix)

        print("=== Total vocab size: ",str(self._count))

    def get_token_to_ix(self,token):
        if token in self.token_to_ix:
            return self.token_to_ix[token]
        else:
            return self.token_to_ix[UNKNOWN_TOKEN]
    
    def get_ix_to_token(self,ix):
        if ix not in self.ix_to_token:
            raise ValueError("Invalid index")
        return self.ix_to_token[ix]
    
    def vocab_size(self):
        return self._count
    
    def __len__(self):
        return self._count


