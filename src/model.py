import torch.nn as nn
import torch
from config import TrainingConfig
from modules.encoders import LanguageEmbeddingLayer, TransformerSequenceEncoder, SinkhornFitter, FusionNet, OutLayer
from modules.OT_solver import OTSolver

class MMAlign(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        attn_dim = config.attn_dim
        nhead = config.nhead
        in_dim1, in_dim2 = config.in_dim1, config.in_dim2
        encoder_dropout = config.encoder_dropout
        num_encoder_layer = config.num_encoder_layer
        num_fusion_layer = config.num_fusion_layer
        snkhrn_rnndim = config.snkhrn_rnndim
        snkhrn_winsize = config.snkhrn_winsize
        output_dropout = config.output_dropout
        self.dataset = config.dataset
        
        self.embedding = LanguageEmbeddingLayer(config)
        self.trse_a = TransformerSequenceEncoder(in_dim1, attn_dim, attn_dim, nhead, encoder_dropout, num_encoder_layer)
        self.trse_b = TransformerSequenceEncoder(in_dim2, attn_dim, attn_dim, nhead, encoder_dropout, num_encoder_layer)
        
        self.sinkhorn = SinkhornFitter(attn_dim, snkhrn_rnndim, snkhrn_winsize)
        self.ot_solver = OTSolver(win_size=snkhrn_winsize)
        self.fusion_network = FusionNet(attn_dim, nhead, num_fusion_layer)
        if self.dataset == 'meld':
            self.out_layer = OutLayer(attn_dim, output_dropout, task='classification')
        else:
            self.out_layer = OutLayer(attn_dim, output_dropout)
    
    def _contra_loss(self, x1, x2):
        # x1 shape [bs, hidden]
        pos = (x1 * x2).sum(-1) # (bs,)
        neg = torch.logsumexp(x1 @ x2.transpose(0, 1), dim=-1)  # (bs, )
        loss = torch.sum(pos - neg)
        return loss
    
    def fit(self, m1, m2, mask, length):
        h1, h2 = self.trse_a(m1, mask), self.trse_b(m2, mask) # (L, B, D)
        A_pred = self.sinkhorn(h1, length)   # (B, L, D)
        A_ot, _ = self.ot_solver(h1.detach(), h2.detach(), length)   # (B, L, D)
        return A_pred, A_ot

    def regression(self, m1, m2, mask):
        h1, h2 = self.trse_a(m1, mask), self.trse_b(m2, mask)
        
        # h1, h2 are (len, bs, hidden)
        h1_head, h2_head = h1[0], h2[0]
        loss_con = self._contra_loss(h1_head, h2_head)
        out = self.out_layer(*self.fusion_network(h1, h2, mask))
        return out, loss_con

    def regression_missing(self, m1, mask, length):
        h1 = self.trse_a(m1, mask)    # (L, B, D)
        A_preds = self.sinkhorn(h1, length)   # (B, L, 2*win_size + 1)
        h2_pred = self.sinkhorn.inference(A_preds, h1.permute(1,0,2), length)  # (B, L, D)
        h2_pred = h2_pred.permute(1, 0, 2) # (L, B, D)
        
        h1_fused, h2_fused = self.fusion_network(h1, h2_pred, mask)
        out = self.out_layer(h1_fused, h2_fused) # (B, 1)
        return out
    
    def forward(self, m1, m2, bert_sent_type, bert_mask, length, mode=0, text_pos=0):
        """
        params:
            @m1 (Tensor): batch of representations of modality 1
            @m2 (Tensor): batch of representations of modality 2
            @mode (int): 
                0 for main task training on the complete partition, 
                1 for sinkhorn MLP fitting with encoder and FusionNet fixed, 
                2 for main task training on missing set and inference on SinkHorn.
            @text_first (bool): a flag indicating whether we need to embed text modality
        returns:
        """
        if self.dataset == 'meld':
            m1 = m1.permute(1, 0, 2) if m1.dim() == 3 else m1.permute(1,0)
            m2 = m2.permute(1, 0, 2) if m2.dim() == 3 else m2.permute(1,0)
            
        modals_in = [m1, m2]
        if text_pos >= 0:
            tp = text_pos
            modals_in[tp] = self.embedding(modals_in[tp])
        
        bert_mask = ~(bert_mask.to(torch.bool))
        assert mode in (0, 1, 2)
        if mode == 0:   # fit OT alignment 
            return self.regression(modals_in[0], modals_in[1], bert_mask)
        elif mode == 1:
            return self.fit(modals_in[0], modals_in[1], bert_mask, length)
        elif mode == 2:    # train on main  
            return self.regression_missing(modals_in[0], bert_mask, length)

if __name__ == '__main__':
    config = TrainingConfig()
    model = MMAlign(config)
    with open('model_info.log', 'w') as f:
        print(model, file=f)
        for n, p in model.named_parameters():
            print(n, file=f)
    total = 0
    for p in model.translator.paramters():
        total += p.numel()
    print("Total Params in Translator: {:10d}".format(total))