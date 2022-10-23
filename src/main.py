import torch
import argparse
import numpy as np
from utils.tools import *
from torch.utils.data import DataLoader
from config import DatasetConfig, TrainingConfig
from solver import Solver
from model import MMAlign
from data_loader import get_loader

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi','mosei','meld'],
                    help='dataset to use (default: mosei)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')
parser.add_argument('--complete_ratio', type=float, default=0.1, 
                    help='complete modality proportion')
parser.add_argument('--modals', type=str, default='tv', help='input modality pair')
parser.add_argument('--group_id', type=int, default=-1, help='training data group')

# Dropouts
parser.add_argument('--encoder_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--output_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--num_encoder_layer', type=int, default=1,
                    help='number of encoder transformer layers')
parser.add_argument('--num_fusion_layer', type=int, default=1,
                    help='number of fusion network layers')

# Sinkhorn
parser.add_argument('--snkhrn_winsize', type=int, default=2, help='window width in sinkhorn solving')

# Losses
parser.add_argument('--lbd', type=float, default=0.0, help='portion of discriminator loss added to total loss (default: 0.1)')

# Architecture
parser.add_argument('--nhead', type=int, default=8,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--attn_dim', type=int, default=32,
                    help='The size of hiddens in all transformer blocks')
parser.add_argument('--snkhrn_rnndim', type=int, default=32,
                    help='dimension of rnn hiddens in sinkhorn solver')


parser.add_argument('--proj_type', type=str, default='cnn',help='network type for input projection', choices=['LINEAR', 'CNN','LSTM','GRU'])

# Tuning
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--warmup_epoch', type=int, default=2, help='warmup epoch for training')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate of main branch (default: 1e-3)')
parser.add_argument('--lr_sinkhorn', type=float, default=1e-3,
                    help='initial learning rate of sinkhorn prediction (default: 1e-3)')
parser.add_argument('--lr_bert', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-3)')

parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--save_name', type=str, default='res', help='file name to save the result')
parser.add_argument('--check_model', action='store_true')

# missing modes (0 for c, and 1 for d)
parser.add_argument('--missing_mode', type=int, choices=[0,1], default=0, help='missing mode (as introduced in paper (0 for c and 1 for d)')

args = parser.parse_args()

# configurations for data_loader
dataset = str.lower(args.dataset.strip())
batch_size = args.batch_size
use_cuda = False
print("Start loading the data....")

train_config = DatasetConfig(dataset, split='train', batch_size=args.batch_size, 
                             data_path=args.data_path, complete_ratio=args.complete_ratio, group_id=args.group_id)
valid_config = DatasetConfig(dataset, split='valid', batch_size=args.batch_size,
                             data_path=args.data_path, complete_ratio=args.complete_ratio if args.missing_mode == 1 else 0.0, group_id=args.group_id)
test_config = DatasetConfig(dataset, split='test',  batch_size=args.batch_size,
                            data_path=args.data_path, complete_ratio=args.complete_ratio if args.missing_mode == 1 else 0.0, group_id=args.group_id)

# print(train_config)
hyp_params = TrainingConfig(vars(args))

# pretrained_emb saved in train_config here
train_complete_loader = get_loader(hyp_params, train_config, mode='complete', shuffle=True)
if hyp_params.complete_ratio < 1.0:
    train_missing_loader = get_loader(hyp_params, train_config, mode='missing', shuffle=True)
else:
    train_missing_loader = None

if args.missing_mode == 1:
    valid_complete_loader = get_loader(hyp_params, valid_config, mode='complete', shuffle=False)
valid_missing_loader = get_loader(hyp_params, valid_config, mode='missing', shuffle=False)

if args.missing_mode == 1:
    test_complete_loader = get_loader(hyp_params, test_config, mode='complete', shuffle=False)
test_missing_loader = get_loader(hyp_params, test_config, mode='missing', shuffle=False)


if dataset == 'meld':
    hyp_params.vocab_size = train_config.vocab_size

if args.missing_mode == 0:
    train_loader = (train_complete_loader, train_missing_loader)
    valid_loader = (None, valid_missing_loader)
    test_loader = (None, test_missing_loader)
else:
    train_loader = (train_complete_loader, train_missing_loader)
    valid_loader = (valid_complete_loader, valid_missing_loader)
    test_loader = (test_complete_loader, test_missing_loader)

print('Finish loading the data....')


if __name__ == '__main__':
    fix_all(args)
    setup_model(hyp_params, train_config, dataset)
    if not hyp_params.check_model:
        solver = Solver(hyp_params, train_loader=train_loader, dev_loader=valid_loader,
                        test_loader=test_loader, is_train=True)
        solver.train_and_eval()
        exit()
    else:
        model = MMAlign(hyp_params)
        with open('model_info.log', 'w') as f:
            print(model, file=f)
            for n, p in model.named_parameters():
                print(n, file=f)
        total = 0
        for p in model.sinkhorn.parameters():
            total += p.numel()
        print("Total Params in Translator: {:10d}".format(total))