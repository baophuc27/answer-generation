import torch.nn as nn
from abc import ABC,abstractmethod

class EncoderBase(nn.Module):
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
