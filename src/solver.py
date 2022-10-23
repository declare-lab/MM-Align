import torch
from torch import nn
import sys
import torch.optim as optim
import numpy as np
import time
import pickle
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# from utils import DiffLoss, MSE, SIMSE, CMD
from utils.eval_metrics import *
from utils.tools import *
from model import MMAlign
from torch.optim import Adam

class Solver(object):
    def __init__(self, hyp_params, train_loader, dev_loader, test_loader, is_train=True, model=None, pretrained_emb=None):
        self.hp = hp = hyp_params
        
        # 't' for text, 'v' for vision, 'a' for acoustic
        assert hp.modals in ['tv','ta','va', 'av', 'at', 'vt']
        self.modals = hp.modals
        
        self.train_loader, self.dev_loader, self.test_loader = train_loader, dev_loader, test_loader
        self.model = MMAlign(hp)
        
        self.clip = 1.0
        
        # Initialize weight of Embedding matrix with Glove embeddings
        if self.hp.pretrained_emb is not None:
            self.model.embedding.embed.weight.data = self.hp.pretrained_emb
        self.model.embedding.embed.requires_grad = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # optimizer
        sinkhorn_params, bert_params, main_params = [], [], []

        self.model.to(self.device)
        self.lbd = hp.lbd
        
        for n, p in self.model.named_parameters():
            # Bert freezing customizations 
            if "bertmodel.encoder.layer" in n:
                layer_num = int(n.split("encoder.layer.")[-1].split(".")[0])
                if layer_num < 8:
                    p.requires_grad = False
                else:
                    bert_params.append(p)
            elif "sinkhorn" in n:
                sinkhorn_params.append(p)
            else:
                main_params.append(p)
                
        self.optimizer_main = Adam(
            [
                {'params':bert_params, 'lr':self.hp.lr_bert},
                {'params':main_params, 'lr':self.hp.lr}
            ]
        )
        self.optimizer_sinkhorn = Adam(
            sinkhorn_params, lr=self.hp.lr_sinkhorn
        )

        # self.criterion_main = nn.MSELoss()
        self.criterion_main = get_main_criterion(hp.dataset)
        self.criterion_alignment_fit = nn.MSELoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=20, factor=0.1, verbose=True)
    
    def _get_input_modal_pairs(self, t, v, a, bert_sent):
        if self.modals == 'tv':
            m1, m2 = t, v
            tp = 0
        elif self.modals == 'ta':
            m1, m2 = t, a
            tp = 0
        elif self.modals == 'va':
            m1, m2 = v, a
            tp = -1
        elif self.modals == 'vt':
            m1, m2 = v, t
            tp = 1
        elif self.modals == 'at':
            m1, m2 = a, t
            tp = 1
        elif self.modals == 'av':
            m1, m2 = a, v
            tp = -1
        
        if self.hp.dataset != 'meld':
            if tp == 0 or tp == -1:
                heading_zeros = torch.zeros(1, m2.size(1), m2.size(2))
                m2 = torch.cat((heading_zeros, m2), dim=0)
            if tp == 1 or tp == -1:
                heading_zeros = torch.zeros(1, m1.size(1), m1.size(2))
                m1 = torch.cat((heading_zeros, m1), dim=0)
        return m1, m2, tp

    def _criterion_fit(self, preds, truths, l):
        """Compute masked MSE Loss
        params:
            @pred (Tensor): prediction of alignment matrix in shape (B, L, win_size)
            @true (Tensor): OT solution of alignment in shape (B, L, L)
        """
        win_size = ((preds.size(-1) - 1) // 2)
        loss = 0.0
        
        for i, (pred, truth) in enumerate(zip(preds, truths)):
            proc_pred = torch.zeros(size=(l[i], l[i]))
            if l[i] > win_size + 1:
                for k in range(-win_size, win_size+1):
                    proc_pred += torch.diagflat(pred[max(-k,0):min(l[i]-k, l[i]), k+win_size], offset=k)
                loss += self.criterion_alignment_fit(proc_pred, truth)
            else:
                proc_pred = torch.cat([pred[k:k+1, win_size-k: win_size-k+l[i]] for k in range(l[i])], dim=0)
                loss += self.criterion_alignment_fit(proc_pred, truth)
        return loss

    def _complete_train_iter(self, batch_data):
        text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch_data
        
        m1, m2, tp = self._get_input_modal_pairs(text, visual, audio, bert_sent)
        preds, l_con = self.model(m1, m2, bert_sent_type, bert_sent_mask, l, mode=0, text_pos=tp)
        return preds, y, l_con
    
    def _complete_fit_iter(self, batch_data):
        text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch_data
        
        m1, m2, tp = self._get_input_modal_pairs(text, visual, audio, bert_sent)
        A_pred, A_ot = self.model(m1, m2, bert_sent_type, bert_sent_mask, l, mode=1, text_pos=tp)
        
        return A_pred, A_ot, l

    def _missing_iter(self, batch_data):
        text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch_data
        
        m1, m2, tp = self._get_input_modal_pairs(text, visual, audio, bert_sent)
        preds = self.model(m1, m2, bert_sent_type, bert_sent_mask, l, mode=2, text_pos=tp)
        
        return preds, y       

    def train(self, epoch):
        self.model.train()
        data_complete_loader, data_missing_loader = self.train_loader
        
        global_step = 0
        
        with tqdm(data_complete_loader) as pbar:
            for batch_data in pbar:
                if epoch > self.hp.warmup_epoch: # at warm-up epoch we do not train translator
                    A_pred, A_truth, l = self._complete_fit_iter(batch_data)
                    
                    fit_loss = self._criterion_fit(A_pred, A_truth, l)
                    self.optimizer_sinkhorn.zero_grad()
                    fit_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer_sinkhorn.step()
                
                preds, y, l_con = self._complete_train_iter(batch_data)
                reg_loss = self.criterion_main(preds, y)
                total_loss = reg_loss - l_con * self.lbd
                self.optimizer_main.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer_main.step()   
                pbar.set_description('Complete Train: Main Loss {:5.4f} | Trans Loss {:5.4f}'.format(reg_loss, fit_loss if epoch > self.hp.warmup_epoch else 0))
                global_step += 1
        
        if epoch > self.hp.warmup_epoch:
            with tqdm(data_missing_loader) as pbar:
                for batch_data in pbar:
                    preds, y = self._missing_iter(batch_data)
                    reg_loss = self.criterion_main(preds, y)
                    self.optimizer_main.zero_grad()
                    reg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer_main.step()
                    pbar.set_description('Missing Train: Main Loss {:5.4f}'.format(reg_loss)) 
                    global_step += 1

    def evaluate(self, test=False):
        self.model.eval()
        # loader = self.test_loader if test else self.dev_loader    
        
        loader_c, loader_m = self.test_loader if test else self.dev_loader
        
        results, truths = [], []
        total_loss, total_l1_loss = 0, 0.0
        total_size = 0

        criterion = self.criterion_main
        criterion_l1 = nn.L1Loss()

        with torch.no_grad():
            with tqdm(loader_m) as pbar:
                for batch in pbar:
                    batch_size = len(batch)
                    preds, y = self._missing_iter(batch)
                    
                    if self.hp.dataset in ['mosi', 'mosei']:
                        total_l1_loss += criterion_l1(preds, y).item() * batch_size
                    total_loss += criterion(preds, y).item() * batch_size
                    
                    total_size += batch_size
                    pbar.set_description('{} Loss {:5.4f}'.format('Evaluating' if not test else 'Testing', total_loss/total_size)) 
                    results.append(preds)
                    truths.append(y)
            
            if self.hp.missing_mode == 1: # dev/test set has modality-complete split
                with tqdm(loader_c) as pbar:
                    for batch in pbar:
                        batch_size = len(batch)
                        preds, y, _ = self._complete_train_iter(batch)
                        if self.hp.dataset in ['mosi', 'mosei']:
                            total_l1_loss += criterion_l1(preds, y).item() * batch_size
                        total_loss += criterion(preds, y).item() * batch_size

                        total_size += batch_size
                        pbar.set_description('{} Reg Loss {:5.4f}'.format('Evaluating' if not test else 'Testing', total_loss/total_size)) 
                        results.append(preds)
                        truths.append(y)                
        
        avg_loss = total_loss / total_size
        avg_l1_loss = total_l1_loss / total_size

        results = torch.cat(results)
        truths = torch.cat(truths)
        
        return avg_loss, avg_l1_loss, results, truths

    def train_and_eval(self):
        best_valid = 1e8
        patience = 20
        best_mae = 1e8

        for epoch in range(self.hp.num_epochs+1):
            start = time.time()
            self.train(epoch)
            
            if epoch > self.hp.warmup_epoch:
                val_loss, val_l1_loss, _, _ = self.evaluate(test=False)
                test_loss, test_l1_loss, results, truths = self.evaluate(test=True)
                end = time.time()
                duration = end-start
                self.scheduler.step(val_loss)    # Decay learning rate by validation loss

                # validation F1
                print("-"*200)
                print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Valid L1 Loss {:5.4f} | Test (L1) Loss {:5.4f}'.format(epoch, duration, val_loss, val_l1_loss, test_l1_loss))
                print("-"*200)

                if best_valid > val_loss:
                    patience = 4 if self.hp.dataset == 'mosei' else 7
                    best_valid = val_loss
                    if test_l1_loss < best_mae:
                        best_mae = test_l1_loss
                        if self.hp.dataset in ["mosei_senti", "mosei"]:
                            eval_res = eval_mosei_senti(results, truths, True)
                        elif self.hp.dataset == 'mosi':
                            eval_res = eval_mosi(results, truths, True)
                        elif self.hp.dataset == 'iemocap':
                            eval_res = eval_iemocap(results, truths)
                    
                        print("Saved model at pre_trained_models/{}_{}.pt!".format(self.hp.dataset, self.hp.complete_ratio))
                        save_model(self.hp, self.model, name=self.hp.dataset)
                
                else:
                    patience -= 1
                    if patience == 0:
                        break
        
        import csv
        try:
            f = open('../results/{}_{}.tsv'.format(self.hp.save_name, self.modals), 'a')
        except:
            f = open('../results/{}_{}.tsv'.format(self.hp.save_name, self.modals), 'w')
        writer = csv.writer(f)
        # if self.hp.complete_ratio == 0.1:
            # writer.writerow(['lbd={}, task={}, group={}, lr={}, lr_trans={}'.format(self.lbd, self.hp.modals, self.hp.group_id, self.hp.lr, self.hp.lr_sinkhorn)])
        writer.writerow(['lbd={}, wrp={},wsize={}, att,nh={},{}, task={}, group={}, lr={}, lr_trans={}'.format(self.hp.lbd, self.hp.warmup_epoch, self.hp.snkhrn_winsize, self.hp.attn_dim, self.hp.nhead, self.hp.modals, self.hp.group_id, self.hp.lr, self.hp.lr_sinkhorn)])
        writer.writerow(['comp ratio'] + list(eval_res.keys()))
        row = [self.hp.complete_ratio]
        for k, v in eval_res.items():
            row.append('{:.3f}'.format(v))
        writer.writerow(row)
        
        sys.stdout.flush()