import random
from unittest.util import _MAX_LENGTH
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import *

from create_dataset import ABCDataset

class MSADataset(Dataset):
    def __init__(self, config, mode):
        ## Fetch dataset
        dataset_name = self.dataset_name = config.dataset.lower()
        assert dataset_name in ['mosi', 'mosei', 'meld']
        dataset = ABCDataset(config)
        
        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.split, mode)

        if dataset_name == 'meld':
            config.visual_size = self.data[0]['video_features'].shape[1]
            config.acoustic_size = self.data[0]['audio_features'].shape[1]
            config.vocab_size = dataset.pretrained_emb.shape[0]
        
        else:
            config.visual_size = self.data[0][0][1].shape[1]
            config.acoustic_size = self.data[0][0][2].shape[1]

        self.len = len(self.data)
        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb

    @property
    def lav_dim(self):
        if self.dataset_name == 'meld':
            return 300, self.data[0]['audio_features'].shape[1], self.data[0]['video_features'].shape[1]
        else:
            return 300, self.data[0][0][2].shape[1], self.data[0][0][1].shape[1]

    @property
    def lav_len(self):
        return 0, 0, 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def get_loader(hp, config, mode='complete', shuffle=True):
    """Load DataLoader of given DialogDataset"""
    dataset = MSADataset(config, mode)
    config.data_len = len(dataset)

    config.lav_dim = dataset.lav_dim
    config.lav_len = dataset.lav_len
    
    if config.split == 'train':
        hp.n_train = len(dataset)
    elif config.split == 'valid':
        hp.n_valid = len(dataset)
    elif config.split == 'test':
        hp.n_test = len(dataset)
        
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def collate_fn(batch):
        if config.dataset == 'meld':
            batch = sorted(batch, key=lambda x: len(x['token_ids']), reverse=True)
            
            labels = [sample['label'] for sample in batch]
            labels = torch.LongTensor(labels)

            text = pad_sequence([torch.LongTensor(sample['token_ids']) for sample in batch],  batch_first=True)
            visual = pad_sequence([torch.FloatTensor(sample['video_features']) for sample in batch], batch_first=True)
            acoustic = pad_sequence([torch.FloatTensor(sample['audio_features']) for sample in batch], batch_first=True)
            
            lengths = torch.LongTensor([len(sample['token_ids']) for sample in batch])

            text = F.pad(text, (1, 0, 0, 0))    # (B, L)
            visual = F.pad(visual, (0, 0, 1, 0, 0, 0))
            acoustic = F.pad(acoustic, (0, 0, 1, 0, 0, 0))  # (B, L, D)
            lengths = lengths + 1

            SENT_LEN = text.size(1)
            
            bert_sent_mask = torch.arange(SENT_LEN).unsqueeze(0).expand_as(text) < lengths.unsqueeze(-1).cuda()

            s, v, a, y, l = text.cuda(), visual.cuda(), acoustic.cuda(), labels.cuda(), lengths.cuda()
            bert_sent_mask = bert_sent_mask.cuda()
            return s, v, a, y, l, None, None, bert_sent_mask
        else:
            # for later use we sort the batch in descending order of length
            batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
            
            # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
            labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
            
            # MOSEI sentiment labels locate in the first column of sentiment matrix
            if labels.size(1) == 7:
                labels = labels[:,0][:,None]

            ## give back to 256
            MAX_LENGTH = 256
            sentences = pad_sequence([torch.LongTensor(sample[0][0][:MAX_LENGTH]) for sample in batch])
            visual = pad_sequence([torch.FloatTensor(sample[0][1][:MAX_LENGTH]) for sample in batch])
            acoustic = pad_sequence([torch.FloatTensor(sample[0][2][:MAX_LENGTH]) for sample in batch])

            SENT_LEN = sentences.size(0)
            # Create bert indices using tokenizer
            lengths = torch.LongTensor([sample[0][0][:MAX_LENGTH].shape[0] for sample in batch])

            bert_details = []
            for sample in batch:
                text = " ".join(sample[0][3])[:MAX_LENGTH]
                encoded_bert_sent = bert_tokenizer.encode_plus(
                    text, max_length=SENT_LEN+2, add_special_tokens=True, truncation=True, padding='max_length')
                bert_details.append(encoded_bert_sent)

            # Bert things are batch_first
            bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
            bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
            bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
            bert_sent, bert_sent_types, bert_sent_mask = bert_sentences.cuda(), bert_sentence_types.cuda(), bert_sentence_att_mask.cuda()

            # lengths are useful later in using RNNs

            s, v, a, y, l = sentences.cuda(), visual.cuda(), acoustic.cuda(), labels.cuda(), lengths.cuda()
            return s, v, a, y, l, bert_sent, bert_sent_types, bert_sent_mask[:,:-1] if bert_sent_mask is not None else None

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return data_loader