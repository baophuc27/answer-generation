import numpy as np
import torch
import time
from core.data.dataset import MyDataset
from torch.utils.data import DataLoader,Dataset

from core.model.net import Net
from core.model.encoder.encoder_lstm import EncoderLSTM
from core.model.decoder.decoder_base import DecoderBase
from core.model.decoder.decoder_lstm import DecoderLSTM
from core.data.utils import count_parameters
from core.model.opt import WarmupOptimizer
class Execution():
    def __init__(self,__C):
        self.__C = __C 
        self.dataset = MyDataset(self.__C)

    def train(self):
        pretrained_emb = torch.FloatTensor(self.dataset.pretrained_emb)
        vocab = self.dataset.vocab
        encoder = EncoderLSTM(pretrained_emb,self.__C)
        decoder = DecoderLSTM(self.__C,pretrained_emb)

        net = Net(encoder,decoder,vocab)
        print("=== Total model parameters: ",count_parameters(net))

        net.train()
        # net.cuda()
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
        dataloader = DataLoader(self.dataset,
                                batch_size=self.__C.BATCH_SIZE,
                                shuffle=True,
                                pin_memory=self.__C.PIN_MEMORY,
                                num_workers = self.__C.NUM_WORKERS)

        for epoch in range(self.__C.MAX_EPOCH):
            loss_epoch = 0
            start_time = time.time()
            
            for step, (
                question_feat,
                answer_feat,
                tgt_feat
            ) in enumerate(dataloader):
                # question_feat.cuda()
                # answer_feat.cuda()
                # tgt_feat.cuda()
                pred = net(question_feat,answer_feat,tgt_feat)
                loss = criterion(pred,tgt_feat.permute((1,0)))
                loss.backward()
                optimizer.step()
                loss_epoch += loss.data.item()
                print("\r[epoch %2d][step %4d/%4d] loss: %.4f" % (
                            epoch + 1,
                            step,
                            int(len(self.dataset) / self.__C.BATCH_SIZE),
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE
                        ), end='          ')

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-start_time)))
        