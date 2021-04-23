import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from core.model.encoder.encoder_lstm import EncoderLSTM
from core.model.decoder.decoder_pgn import DecoderAttention

class PointerGenerator(nn.Module):

    def __init__(self,__C,vocab,encoder, decoder,pretrained_emb):
        super().__init__()
        self.__C = __C
        self.vocab = vocab
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding.from_pretrained(pretrained_emb,freeze=False)

        self.w_h = nn.Linear(self.__C.HIDDEN_DIM*2,1,bias=False)
        self.w_s = nn.Linear(self.__C.HIDDEN_DIM,1,bias=False)
        self.w_x = nn.Linear(self.__C.WORD_EMBED_SIZE,1,bias=True)

        self.beam_size = self.__C.BEAM_SIZE
    
    def forward(self,ques_input,ques_input_text,ans_input,ans_input_text,enc_pad_mask,enc_len,dec_input,max_oov_len):

        enc_hidden , (h, c) = self.encoder(ques_input,ans_input)
        final_dists = []

        # 2. coverage loss - Eq. (12)
        attn_dists = []
        coverages = []
        
        cov = torch.zeros_like(torch.cat([ques_input,ans_input],-1)).float()

        for t in range(self.__C.TGT_MAX_TRAIN):
            input_t = dec_input[:,t] #Decoder input token at timestep t
            vocab_dist, attn_dist, context_vec,h,c = self.decoder(  dec_input=input_t,
                                                                    prev_hidden = h,
                                                                    prev_cell =c,
                                                                    enc_hidden = enc_hidden,
                                                                    enc_pad_mask=enc_pad_mask,
                                                                    coverage = cov)
            cov += attn_dist

            context_feat = self.w_h(context_vec)

            decoder_feat = self.w_s(h)

            input_feat = self.w_x(self.embedding(input_t))

            gen_feat = context_feat + decoder_feat + input_feat

            p_gen = torch.sigmoid(gen_feat)

            vocab_dist = p_gen * vocab_dist                         # [B x V]
            weighted_attn_dist = (1.0 - p_gen) * attn_dist          # [B x L]

            batch_size = vocab_dist.size(0)
            extra_zeros = torch.zeros((batch_size, max_oov_len))
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)  # [B x V_x]

            final_dist = extended_vocab_dist.scatter_add(dim=-1,
                                                         index=enc_input_ext,
                                                         src=weighted_attn_dist)
            # Save outputs for loss computation
            final_dists.append(final_dist)
            attn_dists.append(attn_dist)
            coverages.append(cov)
        
        final_dists = torch.stack(final_dists, dim=-1)  # [B x V_x x T]
        attn_dists = torch.stack(attn_dists, dim=-1)    # [B x L x T]
        coverages = torch.stack(coverages, dim=-1)      # [B x L x T]

        return {
            'final_dist': final_dists,
            'attn_dist': attn_dists,
            'coverage': coverages
        }