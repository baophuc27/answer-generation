from cfgs.base_configs import Configs
import numpy as np
import argparse
from core.utils.preprocess import padding_datasets
import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Answer Generation arguments')

    parser.add_argument('--RUN', dest = 'RUN_MODE'
                        ,choices = ['train','val','test']
                        ,type=str, required=True)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    __C = Configs()
    args = parse_args()
    
    args_dict = __C.parse_to_dict(args)
    
    __C.add_args(args_dict)
    
    __C.proc()

    