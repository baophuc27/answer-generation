import torch.nn as nn
from abc import ABC,abstractmethod

class EncoderBase(nn.Module):
    @abstractmethod
    def __init__(self,pretrained_emb,__C):
        """Constructor of encoder module should take pretrained embedding as 
        an argument because of later comparison of different types of embeddings.

        Args:
            pretrained_emb ([Tensor]): Extracted pretrained embedding.
            __C (object): Config object
        """
        super(EncoderBase,self).__init__()
        self.pretrained_emb = pretrained_emb
        self.__C = __C

    
    @abstractmethod
    def forward(self,question,answer):
        """Base encoder method in full answer generation

        Args:
            question ([Tensor]): Index of questions after tokenized and padded
            answer ([Tensor]): Index of answers after tokenized and padded

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
