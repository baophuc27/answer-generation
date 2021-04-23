import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


class Net(nn.Module):
    def __init__(self,encoder,decoder,vocab):
        super(Net,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
    
    def forward(self,question,answer,target):

        batch_size = target.shape[0]
        target_len = target.shape[1]

        decoder_hidden = self.encoder(question,answer)
        outputs = torch.zeros(target_len,batch_size,self.vocab.__len__())
        input = target[:,0]
        for i in range(1, target_len):
            output, hidden = self.decoder(input,decoder_hidden)
            input = output.argmax(1)
            decoder_hidden= hidden
            outputs[i] = output
        
        return outputs.permute((1,0,2))