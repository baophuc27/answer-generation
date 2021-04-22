import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import math
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from core.model.encoder.encoder_base import EncoderBase

class EncoderLSTM(EncoderBase):
    """ Option 1: Encoder module uses LSTM as a baseline

    Args:
        EncoderBase ([nn.Module]): Abstract class to implement strategy
        design pattern, which may be useful for comparison between
        differents methods.
    """
    def __init__(self,pretrained_emb,__C):
        super(EncoderBase,self).__init__()
        assert pretrained_emb is not None
        self.__C= __C
        self.embedding = nn.Embedding.from_pretrained(pretrained_emb)
        self.embedding.weight.requires_grad = False
        self.embedding.cuda()

        self.lstm_ques = nn.LSTM(input_size=__C.WORD_EMBED_SIZE,
                                hidden_size=__C.HIDDEN_DIM,
                                num_layers=__C.ENCODER_LSTM_LAYERS,
                                batch_first=True,
                                bidirectional=__C.BIDIRECTIONAL_LSTM)

        # LSTM's cell parameters for answer should be different from ques's lstm cell. Can tune them later
        self.lstm_ans = nn.LSTM(input_size=__C.WORD_EMBED_SIZE,
                                hidden_size=__C.HIDDEN_DIM,
                                num_layers=__C.ENCODER_LSTM_LAYERS,
                                batch_first=True,
                                bidirectional=__C.BIDIRECTIONAL_LSTM)

        # self.fc = nn.Linear(in_features=__C.HIDDEN_DIM*num_directions + __C.HIDDEN_DIM*num_directions
        #                     ,out_features=__C.HIDDEN_DIM)
        self.reduce_h = nn.Linear(in_features=2*__C.HIDDEN_DIM,out_features=__C.HIDDEN_DIM)
        self.reduce_c = nn.Linear(in_features=2*__C.HIDDEN_DIM,out_features=__C.HIDDEN_DIM)

        self.dropout = nn.Dropout(p=__C.DROPOUT_RATE)

    def forward(self,question,answer):
        # question.cuda()
        # answer.cuda()
        question_embedding = self.embedding(question)
        answer_embedding = self.embedding(answer)

        _ , (hidden_question,cell_question) = self.lstm_ques(question_embedding)
        
        _ , (hidden_answer,cell_answer) = self.lstm_ans(answer_embedding)

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        hidden = torch.cat([hidden_question,hidden_answer],-1).contiguous()
        
        cell = torch.cat([cell_question,cell_answer],-1).contiguous()

        # hidden = hidden.view(,2*self.__C.HIDDEN_DIM)

        # cell = cell.view(-1,2*self.__C.HIDDEN_DIM)

        hidden = self.dropout(F.relu(self.reduce_h(hidden)))

        cell = self.dropout(F.relu(self.reduce_c(cell)))

        return (hidden,cell)
        
