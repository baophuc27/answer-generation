import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from core.model.encoder.encoder_base import EncoderBase
from core.model.decoder.decoder_base import DecoderBase

class Net(nn.Module):
    def __init__(self,encoder,decoder):
        super(Net,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    