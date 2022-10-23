import sys
import os
import re
import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call, CalledProcessError

import torch
import torch.nn as nn


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# construct a word2id mapping that automatically takes increment when new words are encountered
word2id = defaultdict(lambda: len(word2id))
UNK = word2id['<unk>']
PAD = word2id['<pad>']

# turn off the word2id - define a named function here to allow for pickling
def return_unk():
    return UNK

def load_emb(w2i, path_to_embedding, embedding_size=300, embedding_vocab=2196017, init_emb=None):
    if init_emb is None:
        emb_mat = np.random.randn(len(w2i), embedding_size)
    else:
        emb_mat = init_emb
    f = open(path_to_embedding, 'r')
    found = 0
    for line in tqdm_notebook(f, total=embedding_vocab):
        content = line.strip().split()
        vector = np.asarray(list(map(lambda x: float(x), content[-300:])))
        word = ' '.join(content[:-300])
        if word in w2i:
            idx = w2i[word]
            emb_mat[idx, :] = vector
            found += 1
    print(f"Found {found} words in the embedding file.")
    return torch.tensor(emb_mat).float()

class ABCDataset(object):
    def __init__(self, config):
        DATA_PATH = str(config.dataset_dir)
        if config.dataset.lower() in ['mosi','mosei']:
            CACHE_PATH = DATA_PATH + '/embedding_and_mapping.pt'
        else:
            CACHE_PATH = os.path.join(DATA_PATH, 'embedding.p')
        # If cached data if already exists
        GROUP_DATA_PATH = DATA_PATH + '/resplit'
        GROUP_DATA_PATH += '/group_%d' % config.group_id
        
        if config.split == 'train':
            self.train_complete = load_pickle(GROUP_DATA_PATH + '/train_complete_%d.pkl' % (config.complete_ratio * 100))
            self.train_missing = load_pickle(GROUP_DATA_PATH + '/train_missing_%d.pkl' % (config.complete_ratio * 100))
        
        elif config.split == 'valid':
            if config.complete_ratio > 0:
                self.dev_complete = load_pickle(GROUP_DATA_PATH + '/dev_complete_%d.pkl' % (config.complete_ratio * 100))
                self.dev_missing = load_pickle(GROUP_DATA_PATH + '/dev_missing_%d.pkl' % (config.complete_ratio * 100))
            else:
                self.dev_complete = None
                self.dev_missing = load_pickle(DATA_PATH + '/dev.pkl')
                
        elif config.split == 'test':
            if config.complete_ratio > 0:
                self.test_complete = load_pickle(GROUP_DATA_PATH + '/test_complete_%d.pkl' % (config.complete_ratio * 100))
                self.test_missing = load_pickle(GROUP_DATA_PATH + '/test_missing_%d.pkl' % (config.complete_ratio * 100))
            else:
                self.test_complete = None
                self.test_missing = load_pickle(DATA_PATH + '/test.pkl')
        print('load data from {}: complete!'.format(GROUP_DATA_PATH))
        
        if config.dataset.lower() in ['mosi', 'mosei']:
            self.pretrained_emb, self.word2id = torch.load(CACHE_PATH)
        else:
            self.pretrained_emb = torch.tensor(pickle.load(open(CACHE_PATH, 'rb')))
            self.word2id = None

    def get_data(self, split, mode="complete"):
        if split == "train":
            if mode == "complete": 
                return self.train_complete, self.word2id, self.pretrained_emb
            elif mode == "missing":
                return self.train_missing, self.word2id, self.pretrained_emb
            else:
                raise ValueError('mode can either be "complete" or "missing"')
        elif split == "valid":
            if mode == "complete": 
                return self.dev_complete, self.word2id, self.pretrained_emb
            elif mode == "missing":
                return self.dev_missing, self.word2id, self.pretrained_emb
            else:
                raise ValueError('mode can either be "complete" or "missing"')
        elif split == "test":
            if mode == "complete": 
                return self.test_complete, self.word2id, self.pretrained_emb
            elif mode == "missing":
                return self.test_missing, self.word2id, self.pretrained_emb
            else:
                raise ValueError('mode can either be "complete" or "missing"')
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()