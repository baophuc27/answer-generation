import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import math

from core.model.encoder.encoder import EncoderBase

class EncoderLSTM(EncoderBase):
    """ Option 1: Encoder module uses LSTM as a baseline

    Args:
        EncoderBase ([nn.Module]): Abstract class to implement strategy
        design pattern, which may be useful for comparison between
        differents methods.
    """
    def __init__(self,pretrained_embed):
        super(EncoderBase,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embed)
        
