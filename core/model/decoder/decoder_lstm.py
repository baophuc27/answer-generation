import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from core.model.decoder.decoder_base import DecoderBase

class DecoderLSTM(DecoderBase):
    """Option 1: Decoder uses simple LSTM as a baseline.

    Args:
        DecoderBase ([nn.Module]): [description]
    """
    def __init__(self,__C,pretrained_embedding):
        super(DecoderLSTM,self).__init__()
        self.__C = __C
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        self.embedding.cuda()

        num_directions = 2 if self.__C.BIDIRECTIONAL_LSTM else 1
        if self.__C.DROPOUT_RATE:
            self.dropout = nn.Dropout(self.__C.DROPOUT_RATE)
        self.lstm = nn.LSTM(input_size=__C.WORD_EMBED_SIZE,
                            hidden_size=2*self.__C.ENCODER_HIDDEN_DIM,
                            num_layers=__C.DECODER_LSTM_LAYERS,
                            batch_first=False,
                            bidirectional=__C.BIDIRECTIONAL_LSTM)

        self.combined_size = 2*self.__C.ENCODER_HIDDEN_DIM*num_directions
        self.cover_weight = nn.Parameter(torch.rand(1))
        self.pointer = nn.Linear(self.combined_size,1)

        self.output = nn.Linear(self.combined_size,self.__C.VOCAB_SIZE)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        embedded = self.embedding(input).unsqueeze(0)
        

        if self.__C.DROPOUT_RATE:
            embedded = self.dropout(embedded)

        output, hidden = self.lstm(embedded,hidden)
        
        pointer = self.softmax(self.output(output)).squeeze(0)
        
        return pointer, hidden

        