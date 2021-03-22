import numpy as np
import json, torch, time
from torch.utils import data
from core.data.utils import tokenize


class Dataset(data.Dataset):

    def __init__(self,__C):
        self.__C = __C
        
        self.ques_list = json.load(open(__C.QUESTION_PATH[__C.RUN_MODE],'r'))['questions']
        self.ans_list = json.load(open(__C.ANSWER_PATH[__C.RUN_MODE],'r'))['answers']
        self.tgt_list = json.load(open(__C.TARGET_PATH[__C.RUN_MODE],'r'))['targets']

        self.data_size = self.ques_list.__len__()

        self.all_sent_list = self.ques_list + self.tgt_list
        print("Dataset size: ",self.data_size)
        self.token_to_ix,self.pretrained_emb = tokenize(self.all_sent_list)


        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)
    
    def __getitem__(self,idx):

    def __len__(self):
        return self.data_size
