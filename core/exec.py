import numpy as np
import torch
import torch.nn as nn
import time
import os


from core.data.dataset import MyDataset
from torch.utils.data import DataLoader,Dataset
from core.model.net.pgn import PointerGenerator
from core.model.encoder.encoder_lstm import EncoderLSTM
from core.model.decoder.decoder_lstm import DecoderLSTM
from core.model.decoder.decoder_pgn import DecoderAttention
from core.data.utils import count_parameters
from core.model.opt import WarmupOptimizer
from core.model.loss import Loss
class Execution():
    def __init__(self,__C):
        self.__C = __C 
        self.dataset = MyDataset(self.__C)
        self.loss = Loss(__C)

    def train(self):
        pretrained_emb = torch.FloatTensor(self.dataset.pretrained_emb)
        vocab = self.dataset.vocab
        encoder = EncoderLSTM(pretrained_emb,self.__C)
        decoder = DecoderAttention(self.__C)

        net = PointerGenerator(self.__C,vocab,encoder,decoder)
        print("=== Total model parameters: ",count_parameters(net))

        data_size = len(self.dataset)

        net.train()
        net.cuda()

        # net = nn.DataParallel(net, device_ids=["cuda:0","cuda:1","cuda:2","cuda:3"])

        criterion = self.loss
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                    lr=0.01,
                                    betas=self.__C.OPT_BETAS,
                                    eps=self.__C.OPT_EPS
                                )
        dataloader = DataLoader(self.dataset,
                                batch_size=self.__C.BATCH_SIZE,
                                shuffle=False,
                                pin_memory=self.__C.PIN_MEMORY,
                                num_workers = self.__C.NUM_WORKERS,collate_fn = MyDataset.my_collate)

        os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

        for epoch in range(self.__C.MAX_EPOCH):
            loss_epoch = 0
            start_time = time.time()
            
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()


                question_feat = batch["question_feat"].cuda()
                answer_feat = batch["answer_feat"].cuda()
                tgt_feat = batch["tgt_feat"].cuda()
                
                
                question_text = batch["question_text"]
                answer_text = batch["answer_text"]
                tgt_text = batch["tgt_text"]
                ques_pad_mask = batch["ques_pad_mask"]

                dec_input = tgt_feat[0,:]
                pred = net(question_feat,question_text,answer_feat,answer_text,ques_pad_mask,30,dec_input,5)
                # print("\r[epoch %2d][step %4d/%4d] loss: %.4f" % (
                #             epoch + 1,
                #             step,
                #             int(len(self.dataset) / self.__C.BATCH_SIZE),
                #             loss.cpu().data.numpy()
                #         ), end='          ')

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
            print('Loss of epoch %2d :%.4f' % (epoch_finish,loss_epoch*self.__C.BATCH_SIZE/data_size))
            
    def infer(self):
        pretrained_emb = torch.FloatTensor(self.dataset.pretrained_emb)
        vocab = self.dataset.vocab
        encoder = EncoderLSTM(pretrained_emb,self.__C)
        decoder = DecoderLSTM(self.__C,pretrained_emb)

        net = Net(encoder,decoder,vocab)
        print("=== Total model parameters: ",count_parameters(net))
        state_dict_path = '/home/phuc/Workspace/Thesis/answer-generation/ckpts/ckpt_8294539/epoch10.pkl'
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
            self.infer()
        
        if run_mode == 'test':
            self.train()