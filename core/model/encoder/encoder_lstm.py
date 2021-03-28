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
        self.embedding = nn.Embedding.from_pretrained(pretrained_emb)
        
        num_directions = 2 if __C.BIDIRECTIONAL_LSTM else 1

        self.lstm_ques = nn.LSTM(input_size=__C.WORD_EMBED_SIZE,
                                hidden_size=__C.ENCODER_HIDDEN_DIM,
                                num_layers=__C.ENCODER_LSTM_LAYERS,
                                batch_first=True,
                                bidirectional=__C.BIDIRECTIONAL_LSTM)

        # LSTM's cell parameters for answer should be different from ques's lstm cell. Can tune them later
        self.lstm_ans = nn.LSTM(input_size=__C.WORD_EMBED_SIZE,
                                hidden_size=__C.ENCODER_HIDDEN_DIM,
                                num_layers=__C.ENCODER_LSTM_LAYERS,
                                batch_first=True,
                                bidirectional=__C.BIDIRECTIONAL_LSTM)

        self.fc = nn.Linear(in_features=__C.ENCODER_HIDDEN_DIM*num_directions + __C.ENCODER_HIDDEN_DIM*num_directions
                            ,out_features=__C.DECODER_HIDDEN_DIM)
        
        self.dropout = nn.Dropout(p=__C.DROPOUT_RATE)

    def forward(self,question,answer):
        # question = [question_token_len, batch_size]
        # answer = [answer_token_len, batch_size]
        question_embedding = self.embedding(question)
        answer_embedding = self.embedding(answer)

        outputs_question , hidden_question = self.lstm_ques(question_embedding)

        outputs_answer , hidden_answer = self.lstm_ans(answer_embedding)

        outputs = torch.cat(outputs_question,outputs_answer)

        hidden = torch.tanh(self.fc(torch.cat(hidden_question,hidden_answer)))

        return outputs, hidden
        
