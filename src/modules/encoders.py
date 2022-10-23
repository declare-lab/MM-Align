from unicodedata import bidirectional
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from utils import CMD, MSE
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self, hyp_params):
        super(LanguageEmbeddingLayer, self).__init__()
        self.hp = hp = hyp_params
        if hp.dataset == 'meld':
            self.embed = nn.Embedding(hp.vocab_size, hp.in_dim1)
        else:
            self.embed = nn.Embedding(len(hp.word2id), hp.in_dim1)
        

    def forward(self, sentences, bert_sent=None, bert_sent_type=None, bert_sent_mask=None):
        # extract features from text modality
        output = self.embed(sentences)
        if self.hp.dataset != 'meld':
            heading_zeros = torch.zeros(1, output.size(1), output.size(2)) # (1, B, D)
            output = torch.cat((heading_zeros, output), dim=0)
        return output          

class LinearSequenceEncoder(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim1, out_dim2):
        self.proj1 = nn.Sequential(
            nn.Linear(in_dim1, out_dim1),
            nn.Layernorm(self.out_dim1)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(in_dim2, out_dim2),
            nn.Layernorm(self.out_dim2)          
        )

    def forward(self, m1, m2):
        return self.proj1(m1), self.proj2(m2)

class TransformerSequenceEncoder(nn.Module):
     def __init__(self, in_dim, hidden_dim, out_dim, n_head, dropout, n_layer=1):
         super().__init__()
         self.in_dim = in_dim
         self.proj_layer = nn.Linear(in_dim, hidden_dim)
         single_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_head, dim_feedforward=out_dim, dropout=dropout, activation="gelu")
         self.transformer_encoder = nn.TransformerEncoder(
             single_layer, n_layer
         )
        
     def forward(self, m, mask):
         m = self.proj_layer(m)
         return self.transformer_encoder(m, src_key_padding_mask=mask)

class SequenceEncoder(nn.Module):
    """Encode all modalities with assigned network. The network will output encoded presentations
    of three modalities. The last hidden of LSTM/GRU generates as input to the control module,
    while separate sequence vectors are received by the transformer.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        in_dim1, in_dim2 = config.in_dim1, config.in_dim2
        out_dim1, out_dim2 = config.shared_dim
        
        n_head = config.n_head
        attn_dropout = config.attn_dropout

        self.transformer = TransformerSequenceEncoder(in_dim1, in_dim2, out_dim1, out_dim2, n_head, attn_dropout)
    
    def forward(self, m1, m2):
        """Encode Sequential data from all modalities
        Params:
            @input_l, input_a, input_v (Tuple(Tensor, Tensor)): 
            Tuple containing input and lengths of input. The vectors are in the size 
            (seq_len, batch_size, embed_size)
        Returns:
            All hidden representations are projected to the same size for transformer
            and its controller use.
        """
        return self.transformer.forward(m1, m2)

class SinkhornFitter(nn.Module):
    def __init__(self, in_dim, hid_dim, wsize=0):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hid_dim, bidirectional=True)
        self.out_layer = nn.Sequential(
            nn.Linear(2*hid_dim, 2*wsize+1),
            nn.Softmax(dim=-1)
        )
        assert wsize > 0
        self.wsize = wsize
    
    def forward(self, x, x_lens):
        """Forward the fitting process
        Params:
            @x (Tensor[len, bs, hid_size]): a batch modality sequence representation
            @x_lens (Tensor[bs, hid_size]): corresponding length of modality sequence batch
        """
        packed = pack_padded_sequence(x, x_lens.cpu())
        z, _ = self.rnn(packed) # (l, bs, 2*hid_dim)
        padded, _ = pad_packed_sequence(z, batch_first=True) # (B, L, D)
        align_pred = self.out_layer(padded) # (B, L, winsize)
        return align_pred

    def inference(self, A_preds, x, x_lens):
        """Forward alignment inference process
        params:
            @A (Tensor[B, L, 2*win_size]): alignment matrix
            @x (Tensor[B, L, hid_size]): input modality batch
            @x_lens (Tensor[B,]): input sequence length
        """
        res = []
        B, L, D = x.size()
        for i, (pred, l) in enumerate(zip(A_preds, x_lens)):
            proc_pred = torch.zeros(size=(1, l, l))
            if self.wsize <= l:
                for k in range(-self.wsize, self.wsize+1):
                    proc_pred += torch.diagflat(pred[max(-k,0):min(l-k, l), k+self.wsize], offset=k).unsqueeze(0) # (1, L_x, L_x)
            else:
                proc_pred = torch.stack([pred[ll][self.wsize-ll:self.wsize-ll+l] for ll in range(l)], dim=0).unsqueeze(0) # (1, L_x, L_x)
                
            padding_zeros = torch.zeros(1, L-l, D)
            res.append(torch.cat([torch.matmul(proc_pred, x[i:i+1,:l,:]), padding_zeros], dim=1))
        
        return torch.cat(res, dim=0)

class FusionNet(nn.Module):
    def __init__(self, attn_dim, nhead, num_layers=2):
        super().__init__()
        
        unit_layer = nn.TransformerDecoderLayer(attn_dim, nhead, 4*attn_dim)
        self.a2bfusion = nn.TransformerDecoder(unit_layer, num_layers)
        self.b2afusion = nn.TransformerDecoder(unit_layer, num_layers)
    
    def forward(self, m1, m2, mask):
        a2bres = self.a2bfusion(m1, m2, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
        b2ares = self.b2afusion(m2, m1, tgt_key_padding_mask=mask, memory_key_padding_mask=mask)
        return (a2bres, b2ares)

class OutLayer(nn.Module):
    def __init__(self, attn_dim, dropout, task='regression'):
        super().__init__()
        if task == 'regression':
            self.layers = nn.Sequential(
                nn.Linear(2*attn_dim, 2*attn_dim),
                nn.Tanh(),
                nn.Linear(2*attn_dim, attn_dim),
                nn.Dropout(dropout),
                nn.Linear(attn_dim, 1)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(2*attn_dim, 2*attn_dim),
                nn.Tanh(),
                nn.Linear(2*attn_dim, 7),
            )
    
    def forward(self, m1, m2):
        """Output layer for classification 
        Input:
            @m1 (Tensor): input tensor of modality 1 in shape (L, B, D)
            @m2 (Tensor): input tensor of modality 2 in shape (L, B, D)
        Output:
            @out (Tensor): (B, 1)
        """
        x1, x2 = m1[0,:,:], m2[0,:,:]
        x = torch.cat((x1, x2), dim=-1)
        out = self.layers(x)
        return out