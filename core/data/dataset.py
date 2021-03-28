import numpy as np
import json, torch, time
from torch.utils import data
from core.data.utils import get_pretrained_emb,process_data
from core.data.vocab import Vocab

class Dataset(data.Dataset):

    def __init__(self,__C):
        self.__C = __C
        
        self.ques_list = json.load(open(__C.QUESTION_PATH[__C.RUN_MODE],'r'))['questions']
        self.ans_list = json.load(open(__C.ANSWER_PATH[__C.RUN_MODE],'r'))['answers']
        self.tgt_list = json.load(open(__C.TARGET_PATH[__C.RUN_MODE],'r'))['targets']

        self.data_size = self.ques_list.__len__()
        print("Dataset size: ",self.data_size)

        # Initialize vocab
        all_sent_list = self.ques_list + self.tgt_list
        self.vocab = Vocab(all_sent_list)
        self.pretrained_emb = get_pretrained_emb(self.vocab.ix_to_token)
        
    
    def __getitem__(self,idx):
        ques_feat_iter = np.zeros(1)
        ans_feat_iter = np.zeros(1)
        tgt_feat_iter = np.zeros(1)
        
        ans = self.ans_list[idx]
        ques = self.ques_list[idx]
        tgt = self.tgt_list[idx]
        
        ans_feat_iter = process_data(list(ans.values())[0], self.vocab.token_to_ix, self.__C.ANS_PADDING_TOKEN)
        ques_feat_iter = process_data(list(ques.values())[0], self.vocab.token_to_ix, self.__C.QUES_PADDING_TOKEN)
        tgt_feat_iter = process_data(list(tgt.values())[0], self.vocab.token_to_ix, self.__C.QUES_PADDING_TOKEN)
        
        

        return torch.from_numpy(ques_feat_iter), \
               torch.from_numpy(ans_feat_iter), \
               torch.from_numpy(tgt_feat_iter) 



    def __len__(self):
        return self.data_size
