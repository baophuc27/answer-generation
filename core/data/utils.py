import numpy as np
import en_vectors_web_lg,re,json
from rouge import Rouge

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
            data_idx[ix] = token_to_ix['<unk>']
        
        if ix+1 == max_token:
            break
    
    return data_idx

def insert_sentence_token(features,token_to_ix):
    sos = np.insert(features,0,token_to_ix['<sos>'])
    return np.append(sos,token_to_ix['<eos>'])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval(preds , targets,avg=True):
    rouge = Rouge()
    scores = rouge.get_scores(preds, targets,avg)

    rouge2_f_metric = scores['rouge-2']['f']
    rouge2_p_metric = scores['rouge-2']['p']
    rouge2_r_metric = scores['rouge-2']['r']
    rougel_f_metric = scores['rouge-l']['f']
    rougel_p_metric = scores['rouge-l']['p']
    rougel_r_metric = scores['rouge-l']['r']

    return rouge2_f_metric,rougel_p_metric, \
           rouge2_r_metric,rouge2_f_metric, \
           rouge2_p_metric,rougel_f_metric
           