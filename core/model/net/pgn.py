import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from core.model.encoder.encoder_lstm import EncoderLSTM
from core.model.decoder.decoder_pgn import DecoderAttention

class PointerGenerator(nn.Module):

    def __init__(self,__C,vocab,encoder, decoder):
        super().__init__()
        self.__C = __C
        self.vocab = vocab
        self.encoder = encoder
        self.decoder = decoder

        self.w_h = nn.Linear(self.__C.HIDDEN_DIM*2,1,bias=False)
        self.w_s = nn.Linear(self.__C.HIDDEN_DIM,1,bias=False)
        self.w_x = nn.Linear(self.__C.WORD_EMBED_SIZE,1,bias=True)

        self.beam_size = self.__C.BEAM_SIZE
    
    def forward(self,ques_input,ques_input_text,ans_input,ans_input_text,enc_pad_mask,enc_len,dec_input,max_oov_len):

        enc_hidden, enc_cell = self.encoder(ques_input,ans_input)
        return enc_hidden

