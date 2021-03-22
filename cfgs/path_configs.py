import os

class PATH:
    def __init__(self):
        self.RAW_PATH = './datasets/raw/'
        self.PADDING_PATH = './datasets/padded/'
        self.DATASET_PATH = './datasets/processed/'
        self.init_path()

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
        
        if 'processed' not in os.listdir('./datasets'):
            os.mkdir(('./datasets/processed')) 

        self.QUESTION_PATH = {
            'train' : self.DATASET_PATH + 'train_ques.json',
            'val' : self.DATASET_PATH + 'dev_ques.json',
            'test' : self.DATASET_PATH + 'freebase_test_ques.json'
        }

        self.ANSWER_PATH = {
            'train' : self.DATASET_PATH + 'train_ans.json',
            'val' : self.DATASET_PATH + 'dev_ans.json',
            'test' : self.DATASET_PATH + 'freebase_test_ans.json'
        }

        self.TARGET_PATH = {
            'train' : self.DATASET_PATH + 'train_tgt.json',
            'val' : self.DATASET_PATH + 'dev_tgt.json',
            'test' : self.DATASET_PATH + 'freebase_test_tgt.json'
        }

        def check_path(self):
            print('Checking dataset ...')

            for mode in self.TARGET_PATH:
                if not os.path.exists(self.TARGET_PATH[mode]):
                    print(self.TARGET_PATH[mode] + 'NOT EXIST')
                    exit(-1)

            for mode in self.QUESTION_PATH:
                if not os.path.exists(self.QUESTION_PATH[mode]):
                    print(self.QUESTION_PATH[mode] + 'NOT EXIST')
                    exit(-1)

            for mode in self.ANSWER_PATH:
                if not os.path.exists(self.ANSWER_PATH[mode]):
                    print(self.ANSWER_PATH[mode] + 'NOT EXIST')
                    exit(-1)

            print('Finished')
            print('')
