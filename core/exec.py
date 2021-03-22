import numpy as np
import torch
from core.data.dataset import Dataset

class Execution():
    def __init__(self,__C):
        self.__C = __C 
        self.dataset = Dataset(self.__C)
