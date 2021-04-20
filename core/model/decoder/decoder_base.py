from abc import ABC,abstractmethod
import torch.nn as nn

class DecoderBase(nn.Module):
    @abstractmethod
    def __init__(self):
        """ Base decoder module for inheritation.

        Args:
            __C (object): Config object
        """
        super(DecoderBase,self).__init__()

    @abstractmethod
    def forward(self, input, hidden,cell):
        """Base decoder in full answer generation

        Args:
            context ([Tensor]): Features after decoder module

        Raises:
            NotImplementedError
        """
        raise NotImplementedError