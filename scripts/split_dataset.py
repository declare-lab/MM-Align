import os
import re
import pickle
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call, CalledProcessError

import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description='Data to Split')
parser.add_argument('--complete_ratio', type=float, default=0.2)
parser.add_argument('--data_path', type=str, default='data')
parser.add_argument('--seed', type=int, default=2022, help='random seed to sample complete modalities')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--group_id', type=int, default=1)
args = parser.parse_args()

np.random.seed(args.seed)

def split():
    DATA_PATH = str(args.data_path)
    TRAIN_SET_PATH = DATA_PATH + '/{}.pkl'.format(args.split)
    
    if not os.path.exists(TRAIN_SET_PATH):
        exit('Not find target cache file %s' % TRAIN_SET_PATH)
        
    with open(TRAIN_SET_PATH, 'rb') as f:
        train_set = pickle.load(f)
        f.close()
    
    num_data = len(train_set)
    num_to_sample = np.int(np.floor(num_data * args.complete_ratio))
    samp_inds = np.random.choice(num_data, num_to_sample, replace=False)
     
    # extract complete samples
    train_complete_set = []
    for ind in samp_inds:
        train_complete_set.append(train_set[ind])
    
    # extract missing samples
    unsamp_inds = list(set(i for i in range(num_data)) - set(samp_inds))
    train_missing_set = []
    for ind in unsamp_inds:
        train_missing_set.append(train_set[ind])
    
    if not os.path.exists(DATA_PATH + '/resplit/group_{}'.format(args.group_id)):
        os.mkdir(DATA_PATH + '/resplit/group_{}'.format(args.group_id))
    
    TRAIN_COMPLETE_SAVE_PATH = DATA_PATH + '/resplit/group_%d/%s_complete_%d.pkl' % (args.group_id, args.split, 100*args.complete_ratio)
    with open(TRAIN_COMPLETE_SAVE_PATH, 'wb') as f:
        pickle.dump(train_complete_set, f)
        f.close()
        
    TRAIN_MISSING_SAVE_PATH = DATA_PATH + '/resplit/group_%d/%s_missing_%d.pkl' % (args.group_id, args.split, 100*args.complete_ratio)
    with open(TRAIN_MISSING_SAVE_PATH, 'wb') as f:
        pickle.dump(train_missing_set, f)
        f.close()
    
    print('Finished splitting: complete length {}, missing length {}'.format(len(train_complete_set), len(train_missing_set)))

if __name__ == '__main__':
    split()