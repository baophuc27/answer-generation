import numpy as np
import en_vectors_web_lg,re,json

def tokenize(all_sent_list):
    token_to_ix ={
        'PAD': 0,
        'UNK': 1,
    }
    glove = en_vectors_web_lg.load()
    pretrained_emb = []
    pretrained_emb.append(glove('PAD').vector)
    pretrained_emb.append(glove('UNK').vector)

    i=0
    for item in all_sent_list:
        sent = list(item.values())[0]
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            sent.lower()
        ).replace('-', ' ').replace('/', ' ').split()
        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                pretrained_emb.append(glove(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb

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
