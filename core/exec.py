import numpy as np
import torch
from core.data.dataset import Dataset
from core.model.net import Net
from core.model.encoder.encoder_lstm import EncoderLSTM
from core.model.decoder.decoder_base import DecoderBase
from core.data.utils import count_parameters

class Execution():
    def __init__(self,__C):
        self.__C = __C 
        self.dataset = Dataset(self.__C)

    def train(self):
        pretrained_emb = torch.FloatTensor(self.dataset.pretrained_emb)
        vocab = self.dataset.vocab
        encoder = EncoderLSTM(pretrained_emb,self.__C)
        decoder = DecoderBase(self.__C)

        net = Net(encoder,decoder)
        print("=== Total model parameters: ",count_parameters(net))