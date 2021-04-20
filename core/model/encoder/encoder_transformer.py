import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import math
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from core.model.encoder.encoder_base import EncoderBase

class EncoderTransformer(EncoderBase):

    def __init__(self,pretrained_emb,__C):
        super(EncoderBase,self).__init__()
        assert pretrained_emb is not None
        

    def forward(self,question,answer):
        pass
        
