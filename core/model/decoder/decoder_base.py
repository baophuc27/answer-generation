from abc import ABC,abstractmethod
import torch.nn as nn

class DecoderBase(nn.Module):
    @abstractmethod
    def __init__(self,__C):
        """ Base decoder module for inheritation.

        Args:
            __C (object): Config object
        """
        super(DecoderBase,self).__init__()
        self.__C = __C

    @abstractmethod
    def forward(self,embedding):
        """Base decoder in full answer generation

        Args:
            embedding ([Tensor]): Features after decoder module

        Raises:
            NotImplementedError
        """
        raise NotImplementedError