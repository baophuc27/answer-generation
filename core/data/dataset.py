import numpy as np
import json, torch, time
from torch.utils import data
from core.data.utils import get_pretrained_emb,segment,insert_sentence_token,collate_tokens
from core.data.vocab import Vocab
import functools
import operator
class MyDataset(data.Dataset):

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
        assert self.vocab.vocab_size() > 0
        setattr(__C,'VOCAB_SIZE',self.vocab.vocab_size())

        self.pretrained_emb = torch.FloatTensor(get_pretrained_emb(self.vocab.total_ix_to_token))
        
    
    def __getitem__(self,idx):
        
        ans = self.ans_list[idx]
        ques = self.ques_list[idx]
        tgt = self.tgt_list[idx]
        
        answer_tokens = segment(list(ans.values())[0])
        ques_tokens = segment(list(ques.values())[0])
        tgt_tokens = segment(list(tgt.values())[0])
        # ans_feat_iter = process_data(list(ans.values())[0], self.vocab.token_to_ix, self.__C.ANS_PADDING_TOKEN)
        # ques_feat_iter = process_data(list(ques.values())[0], self.vocab.token_to_ix, self.__C.QUES_PADDING_TOKEN)
        # tgt_feat_iter = process_data(list(tgt.values())[0], self.vocab.token_to_ix, self.__C.QUES_PADDING_TOKEN)
        ques_feat,ans_feat, oovs = self.vocab.source2ids_extend(ques_tokens,answer_tokens)
        tgt_ids = [self.vocab.get_token_to_ix(token) for token in tgt_tokens]
        tgt_ids_ext = self.vocab.target2ids_extend(tgt_tokens,oovs)
        

        return {
                "question_text":ques,
                "answer_text":ans,
                "tgt_text":tgt,
                "ques_feat": ques_feat,
                "ans_feat": ans_feat,
                "tgt_feat":tgt_ids,
                "tgt_feat_ext":tgt_ids_ext,
                "oovs": oovs
                }

    @staticmethod
    def my_collate(batch):
        pad_id = 3
        start_id = 0
        end_id = 1
        ques = [item["question_text"] for item in batch]
        ans = [item["answer_text"] for item in batch]
        tgt = [item["tgt_text"] for item in batch]
        list_oovs = [item["oovs"] for item in batch]

        oovs = functools.reduce(operator.iconcat, list_oovs,[])
        max_oov_len = max([len(item["oovs"]) for item in batch])
        enc_len = torch.LongTensor([len(item["ques_feat"]) for item in batch])

        question_feat = collate_tokens([item["ques_feat"] for item in batch],pad_idx = pad_id)
        answer_feat = collate_tokens([item["ans_feat"] for item in batch],pad_idx = pad_id)
        tgt_feat = collate_tokens([item["tgt_feat"] for item in batch],pad_idx = pad_id)

        start_token = torch.empty(question_feat.shape[0]).fill_(start_id).unsqueeze(1)
        end_token = torch.empty(question_feat.shape[0]).fill_(end_id).unsqueeze(1)

        question_feat = torch.cat([start_token,question_feat,end_token],dim=1).long()
        answer_feat = torch.cat([start_token,answer_feat,end_token],dim=1).long()
        tgt_feat = torch.cat([start_token,tgt_feat,end_token],dim=1).long()

        ques_pad_mask = (question_feat == pad_id)


        return {
            "question_text":ques,
            "answer_text":ans,
            "tgt_text":tgt,
            "question_feat" : question_feat,
            "answer_feat" : answer_feat,
            "tgt_feat" : tgt_feat,
            "ques_pad_mask": ques_pad_mask,
            "oovs": oovs,
            "max_oov_len":max_oov_len,
            "enc_len":enc_len
        }

    def __len__(self):
        return self.data_size
