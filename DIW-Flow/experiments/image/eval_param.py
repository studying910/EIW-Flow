import os
import math

import numpy as np
import torch
import pickle
import argparse
import torchvision.utils as vutils

# Data
from data.data import get_data, get_data_id, add_data_args

# Model
from model.model_flow import get_model, get_model_id, add_model_args
from denseflow.distributions import DataParallelDistribution

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--double', type=eval, default=False)
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)  # the same seed produces the same result

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

##################
## Specify data ##
##################

_, _, data_shape, _ = get_data(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check)
model.load_state_dict(checkpoint['model'])
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

############
## Sample ##
############

path_param = '{}/param/param.txt'.format(eval_args.model, checkpoint['current_epoch'], eval_args.seed)
if not os.path.exists(os.path.dirname(path_param)):
    os.mkdir(os.path.dirname(path_param))

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.train()
if eval_args.double: model = model.double()

param_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
with open(path_param, 'w') as f:
    f.write(str(param_size))
