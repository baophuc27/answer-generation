from abc import ABC,abstractmethod
import torch.nn as nn

class DecoderBase(nn.Module):
    @abstractmethod
    def forward(self,embedding):
        """Base decoder in full answer generation

        Args:
            embedding ([Tensor]): Features after decoder module

        Raises:
            NotImplementedError
        """
        raise NotImplementedError