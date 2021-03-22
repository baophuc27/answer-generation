import os

class PATH:
    def __init__(self):
        self.RAW_PATH = './datasets/raw/'
        self.PADDING_PATH = './datasets/padded/'
        self.DATASET_PATH = './datasets/processed/'

    def init_path(self):
        self.PRED_PATH = './results/pred/'
        self.CACHE_PATH = './results/cache/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './ckpts/'

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')
        
        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')
        
        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')

