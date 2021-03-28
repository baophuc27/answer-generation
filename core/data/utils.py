import numpy as np
import en_vectors_web_lg,re,json

def get_pretrained_emb(all_tokens):
    glove = en_vectors_web_lg.load()
    pretrained_emb = []
    for token_ix in all_tokens:
        pretrained_emb.append(glove(all_tokens[token_ix]).vector)

    return pretrained_emb

def process_data(ques,token_to_ix,max_token):
    data_idx = np.zeros(max_token, np.int64)

    words = re.sub(r"([.,'!?\"()*#:;])",
        '',ques.lower()).replace('-',' ').replace('/',' ').split()
    
    for ix,word in enumerate(words):
        if word in token_to_ix:
            data_idx[ix] = token_to_ix[word]
        else:
            data_idx[ix] = token_to_ix['UNK']
        
        if ix+1 == max_token:
            break
    
    return data_idx
