
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,__C):
        super().__init__()
        self.__C = __C
        self.v = nn.Linear(in_features=2*self.__C.HIDDEN_DIM,out_features=1,bias=False)
        self.enc_proj = nn.Linear(in_features = 2*self.__C.HIDDEN_DIM,out_features=2*self.__C.HIDDEN_DIM,bias=False)
        self.dec_proj = nn.Linear(in_features = self.__C.HIDDEN_DIM,out_features=2.self.__C.HIDDEN_DIM,bias=True)
        
        if self.__C.USE_COVERAGE:
            self.w_c = nn.Linear(1,self.__C.HIDDEN_DIM,bias=False)
    
    def forward(self,decoder_input,encoder_hidden,coverage,enc_pad_mask):
        enc_feature = self.enc_proj(encoder_hidden)
        dec_feature = self.dec_proj(decoder_input)

        scores = torch.tanh(enc_feature + dec_feature)
        scores = self.v(scores)
        
        if self.__C.USE_COVERAGE:
            coverage = coverage.unsqueeze(-1)
            cov_feature = self.w_c(coverage)
            scores += cov_feature
        
        if enc_pad_mask is not None:
            scores = scores.float().masked_fill_(
                enc_pad_mask,
                float('-inf')
            ).type_as(scores)  

        attn_dist = F.softmax(scores,dim=-1)
        return attn_dist

class DecoderAttention(nn.Module):
    def __init__(self,__C):
        super().__init__()
        self.__C = __C

        self.lstm = nn.LSTM(input_size = __C.WORD_EMBED_SIZE,
                            hidden_size = __C.HIDDEN_DIM,
                            bidirectional=False)
        self.attention = Attention(self.__C)

        self.v = nn.Linear(__C.HIDDEN_DIM*3,__C.HIDDEN_DIM,bias=True)
        self.v_out = nn.Linear(__C.HIDDEN_DIM,__C.VOCAB_SIZE,bias=True)
    
    
    def forward(self,dec_input,prev_hidden,prev_cell,enc_hidden,enc_pad_mask,coverage):
        
        hidden,cell = self.lstm(dec_input,(prev_hidden,prev_cell))

        attn_dist = self.attention(dec_input,enc_hidden,coverage,enc_pad_mask)

        context_vec = torch.matmul(attn_dist.unsqueeze(1),enc_hidden)

        context_vec = torch.sum(context_vec,dim =1)

        output = self.v(torch.cat([hidden,context_vec],dim=-1))

        output = self.v_out(output)

        vocab_dist = F.softmax(output,dim=-1)

        return vocab_dist,attn_dist,context_vec,hidden,cell





        
