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

        self.total_token_to_ix = {}
        self.total_ix_to_token = {}

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
        filter_tokens =  list(filter(lambda x : counter[x] > 3 , counter))
        all_tokens = list(counter)


        for token in [PAD_TOKEN,UNKNOWN_TOKEN,STOP_DECODING,START_DECODING]:
            filter_tokens.insert(0,token)
            all_tokens.insert(0,token)

        for ix,token in enumerate(all_tokens):
            self.total_token_to_ix[token] = ix
            self.total_ix_to_token[ix] = token

        # Dictionary for tokens in vocab
        for ix,token in enumerate(filter_tokens):
            self.token_to_ix[token] = ix
            self.ix_to_token[ix] = token
        self._count = len(self.token_to_ix)

        print("=== Total vocab size: ",str(self._count))
        print("=== Total token in dataset: ",str(len(self.total_token_to_ix)))
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
    
    def source2ids_extend(self,ques_tokens,ans_tokens):
        ques_ids = []
        ans_ids = []
        oovs = []
        for ques_token in ques_tokens :
            t_id = self.get_token_to_ix(ques_token)
            unk_id = self.get_token_to_ix(UNKNOWN_TOKEN)
            if t_id == unk_id:
                if ques_token not in oovs:
                    oovs.append(ques_token)
                ques_ids.append(self.__len__() + oovs.index(ques_token))
            else:
                ques_ids.append(t_id)
        
        for ans_token in ans_tokens :
            t_id = self.get_token_to_ix(ans_token)
            unk_id = self.get_token_to_ix(UNKNOWN_TOKEN)
            if t_id == unk_id:
                if ans_token not in oovs:
                    oovs.append(ans_token)
                ans_ids.append(self.__len__() + oovs.index(ans_token))
            else:
                ans_ids.append(t_id)

        return ques_ids,ans_ids,oovs
                
    def target2ids_extend(self,tgt_tokens,oovs):
        ids = []
        for token in tgt_tokens:
            t_id = self.get_token_to_ix(token)
            unk_id = self.get_token_to_ix(UNKNOWN_TOKEN)
            if t_id == unk_id:
                if token in oovs:
                    ids.append(self.__len__() + oovs.index(token))
                else:
                    ids.append(unk_id)
            else:
                ids.append(t_id)
        return ids


