import numpy as np
import torch
import torch.nn as nn
import time
import os


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
        net.cuda()

        net = nn.DataParallel(net, device_ids=["cuda:0","cuda:1","cuda:2","cuda:3"])

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                    lr=0.01,
                                    betas=self.__C.OPT_BETAS,
                                    eps=self.__C.OPT_EPS
                                )
        dataloader = DataLoader(self.dataset,
                                batch_size=self.__C.BATCH_SIZE,
                                shuffle=False,
                                pin_memory=self.__C.PIN_MEMORY,
                                num_workers = self.__C.NUM_WORKERS)

        os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

        for epoch in range(self.__C.MAX_EPOCH):
            loss_epoch = 0
            start_time = time.time()
            
            for step, (
                question_feat,
                answer_feat,
                tgt_feat
            ) in enumerate(dataloader):
                question_feat = question_feat.cuda()
                answer_feat = answer_feat.cuda()
                tgt_feat = tgt_feat.cuda()
                optimizer.zero_grad()
                pred = net(question_feat,answer_feat,tgt_feat)
                output_dim = pred.shape[-1]
                # tgt_feat = tgt_feat.T
                pred = pred.contiguous().view(-1,output_dim).cuda()
                tgt_feat = tgt_feat.T.contiguous().view(-1)
                loss = criterion(pred,tgt_feat)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.data.item()
                print("\r[epoch %2d][step %4d/%4d] loss: %.4f" % (
                            epoch + 1,
                            step,
                            int(len(self.dataset) / self.__C.BATCH_SIZE),
                            loss.cpu().data.numpy()
                        ), end='          ')

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-start_time)))

            epoch_finish = epoch +1 
            torch.save(
                net.state_dict(),
                self.__C.CKPTS_PATH +
                'ckpt_' + self.__C.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

    def infer(self):
        pretrained_emb = torch.FloatTensor(self.dataset.pretrained_emb)
        vocab = self.dataset.vocab
        encoder = EncoderLSTM(pretrained_emb,self.__C)
        decoder = DecoderLSTM(self.__C,pretrained_emb)

        net = Net(encoder,decoder,vocab)
        print("=== Total model parameters: ",count_parameters(net))
        state_dict_path = '/home/phuc/Workspace/Thesis/answer-generation/ckpts/ckpt_9177283/epoch10.pkl'
        net = nn.DataParallel(net, device_ids=["cuda:0","cuda:1","cuda:2","cuda:3"])
        state_dict = torch.load(state_dict_path)
        net.load_state_dict(torch.load(state_dict_path))
        net.eval()
        net.cuda()

        

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                    lr=0.01,
                                    betas=self.__C.OPT_BETAS,
                                    eps=self.__C.OPT_EPS
                                )
        dataloader = DataLoader(self.dataset,
                                batch_size=self.__C.BATCH_SIZE,
                                shuffle=True,
                                pin_memory=self.__C.PIN_MEMORY,
                                num_workers = self.__C.NUM_WORKERS)

        os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

        for epoch in range(self.__C.MAX_EPOCH):
            loss_epoch = 0
            start_time = time.time()
            
            for step, (
                question_feat,
                answer_feat,
                tgt_feat
            ) in enumerate(dataloader):
                question_feat = question_feat.cuda()
                answer_feat = answer_feat.cuda()
                tgt_feat = tgt_feat.cuda()
                optimizer.zero_grad()
                preds = net(question_feat,answer_feat,tgt_feat)
                # tgt_feat = tgt_feat.T
                preds = preds.cpu().data.numpy()
                for pred in preds:
                    out = np.argmax(pred,axis=1)
                    sent = [self.dataset.vocab.get_ix_to_token(index) for index in out]
                    print(' '.join(sent))
                return

    def run(self,run_mode):
        if run_mode == 'train':
            self.train()
        
        if run_mode == 'test':
            self.train()