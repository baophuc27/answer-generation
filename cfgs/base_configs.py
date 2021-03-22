import os, torch, random
import numpy as np 
from types import MethodType
from cfgs.path_configs import PATH

class Configs(PATH):
    def __init__(self):
        super(Configs,self).__init__()

        self.GPU = '0'

        self.SEED = random.randint(0,9999999)

        self.VERSION = str(self.SEED)

        # --------------------------------------------------
        # --------------- MODEL PARAM ----------------------
        # --------------------------------------------------

        self.WORD_EMBED_SIZE = 768

        self.PADDING_TOKEN = 30

        self.BATCH_SIZE = 128
    
    def parse_to_dict(self,args):
        args_dict = {}
        for arg in dir(args):
            print(arg)

    