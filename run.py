from cfgs.base_configs import Configs
import numpy as n
import argparse
from core.utils.preprocess import padding_datasets

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
    
    __C.parse_to_dict(args)