import torch
import random
import numpy as np
import argparse


def training_presetting():
    random.seed(42)
    # torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='STGCN, for centralized and federated model training')
    parser.add_argument('--enable_cuda', default='True', help='Enable CUDA')
    args = parser.parse_args()
    args.device = None
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    return args
