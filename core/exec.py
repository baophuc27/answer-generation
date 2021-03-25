import numpy as np
import torch
from core.data.dataset import Dataset

class Execution():
    def __init__(self,__C):
        self.__C = __C 
        self.dataset = Dataset(self.__C)

    def train(self):
        datasize = self.dataset.data_size
        for i in range(5):
            ques,_,_ = self.dataset[i]
            print(type(ques))