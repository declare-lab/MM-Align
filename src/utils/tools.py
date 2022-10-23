from numpy import diagonal
import torch
import os

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'meld': 7
}

def save_model(args, model, name=''):
    if not name:
        name = 'best'
    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')
    torch.save(model, f'pre_trained_models/{name}.pt')

def load_model(args, name=''):
    if not name:
        name = 'best'
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model

def nopeak_mask(size, l, max_l):
    ones = torch.ones(1, size, l).astype('uint8') # (1, size, max_len)
    zeros = torch.zeros(1, size, max_l -l).astype('unit8') # (1, size, max_len - len)
    np_mask = torch.cat(ones, zeros, dim=2)
    np_mask = np_mask.cuda()
    return np_mask

def fix_all(args):
    torch.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        if args.no_cuda:
            print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        else:
            torch.cuda.manual_seed_all(args.seed)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            use_cuda = True

def setup_model(hyp_params, train_config, dataset):
    # addintional appending
    hyp_params.word2id = train_config.word2id
    hyp_params.pretrained_emb = train_config.pretrained_emb

    # architecture parameters
    hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_config.lav_dim
        
    if hyp_params.modals == 'tv':
        hyp_params.in_dim1 = hyp_params.orig_d_l
        hyp_params.in_dim2 = hyp_params.orig_d_v
    elif hyp_params.modals == 'ta':
        hyp_params.in_dim1 = hyp_params.orig_d_l
        hyp_params.in_dim2 = hyp_params.orig_d_a
    elif hyp_params.modals == 'va':
        hyp_params.in_dim1 = hyp_params.orig_d_v
        hyp_params.in_dim2 = hyp_params.orig_d_a
    elif hyp_params.modals == 'av':
        hyp_params.in_dim1 = hyp_params.orig_d_a
        hyp_params.in_dim2 = hyp_params.orig_d_v
    elif hyp_params.modals == 'at':
        hyp_params.in_dim1 = hyp_params.orig_d_a
        hyp_params.in_dim2 = hyp_params.orig_d_l
    elif hyp_params.modals == 'vt':
        hyp_params.in_dim1 = hyp_params.orig_d_v
        hyp_params.in_dim2 = hyp_params.orig_d_l
        
    hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_config.lav_len
    hyp_params.dataset = hyp_params.data = dataset
    hyp_params.output_dim = output_dim_dict.get(dataset, 1)