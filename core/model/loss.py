import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self,__C):
        self.use_coverage  = __C.USE_COVERAGE
        self.cov_weight = __C.COV_WEIGHT
        self.pad_id = 2
    
    def nll_loss(self,output,target):
        output = torch.log(output)
        return F.nll_loss(output,target,ignore_index=self.pad_id,reduction='mean')
    
    def cov_loss(self,attn_dist,coverage,dec_pad_mask,dec_len):
        min_val = torch.min(attn_dist, coverage)    # [B x L x T]
        loss = torch.sum(min_val, dim=1)            # [B x T]

        # ignore loss from [PAD] tokens
        loss = loss.masked_fill_(
            dec_pad_mask,
            0.0
        )
        avg_loss = torch.sum(loss) / torch.sum(dec_len)
        return avg_loss

    def forward(self,output,batch):
        final_dist = output['final_dist']
        dec_target = batch.dec_target
        nll_loss = self.nll_loss(output=final_dist, target=dec_target)

        attn_dist = output['attn_dist']
        coverage = output['coverage']
        dec_pad_mask = batch.dec_pad_mask
        dec_len = batch.dec_len
        cov_loss = self.cov_loss(attn_dist, coverage, dec_pad_mask, dec_len)
        return nll_loss, cov_loss
    
    