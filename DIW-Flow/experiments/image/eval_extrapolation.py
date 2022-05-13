import os
import math

import numpy as np
import torch
import pickle
import argparse
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import time

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
parser.add_argument('--numbers', type=int, default=10)  # number of picture to reconstruct
parser.add_argument('--t', type=float, default=1.0)  # temperature-annealing
parser.add_argument('--choice', type=str, default='two_points_single_extra',
                    choices=['two_points_single_extra', 'two_points_multiple_extra', 'two_classes', 'recon_extra',
                             'extra_recon',
                             'extra_recon_class', 'recon_extra_class',
                             'reconstruction'])  # selection of sampling method
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)  # the same seed produces the same result

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

args.name = time.strftime("%Y-%m-%d_%H-%M-%S")

##################
## Specify data ##
##################

train_loader, eval_loader, data_shape, dataset = get_data(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check)
model.load_state_dict(checkpoint['model'])
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()
if eval_args.double: model = model.double()

############
## Sample ##
############

# extrapolate in latent space but only in the smallest scale
if eval_args.choice == 'two_points_single_extra':

    path_samples = '{}/two_points_single_extra/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                                           eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    samples = model.two_points_single_extra(t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)
    path_samples_t = '{}/two_points_single_extra/ep{}_s{}_t{}_{}.png'.format(eval_args.model,
                                                                             checkpoint['current_epoch'],
                                                                             eval_args.seed, eval_args.t, args.name)
    vutils.save_image(samples, fp=path_samples_t, nrow=9)

# extrapolate in the latent space with multi-scale
elif eval_args.choice == 'two_points_multiple_extra':
    path_samples = '{}/two_points_multiple_extra/sample_ep{}_s{}.png'.format(eval_args.model,
                                                                             checkpoint['current_epoch'],
                                                                             eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    samples = model.two_points_multiple_extra(t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)
    path_samples_t = '{}/two_points_multiple_extra/ep{}_s{}_t{}_{}.png'.format(eval_args.model,
                                                                               checkpoint['current_epoch'],
                                                                               eval_args.seed, eval_args.t, args.name)
    vutils.save_image(samples, fp=path_samples_t, nrow=9)

# first extrapolate in X, then forward to Z, and extrapolate in Z with the sample extrapolation method, and backward
# to X
elif eval_args.choice == 'recon_extra':
    path_samples = '{}/recon_extra/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                               eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    path_samples_time = '{}/recon_extra/{}/sample_ep{}_s{}.png'.format(eval_args.model, args.name,
                                                                       checkpoint['current_epoch'],
                                                                       eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples_time)):
        os.mkdir(os.path.dirname(path_samples_time))

    # Data Loader
    train_loader_recon_extra = DataLoader(dataset.train, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                          pin_memory=args.pin_memory)
    eval_loader_recon_extra = DataLoader(dataset.test, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                         pin_memory=args.pin_memory)

    train_i = 1
    x1_train = None
    x2_train = None
    for x in train_loader_recon_extra:
        x = x.to(device)
        if train_i == 1:
            x1_train = x
            train_i = train_i + 1
        elif train_i == 2:
            x2_train = x
            train_i = train_i + 1
        else:
            break

    eval_i = 1
    x1_eval = None
    x2_eval = None
    for y in eval_loader_recon_extra:
        y = y.to(device)
        if eval_i == 1:
            x1_eval = y
            eval_i = eval_i + 1
        elif eval_i == 2:
            x2_eval = y
            eval_i = eval_i + 1
        else:
            break

    train_samples = model.forward_recon_extra(x1_train, x2_train, t=eval_args.t).cpu().float() / (
            2 ** args.num_bits - 1)
    eval_samples = model.forward_recon_extra(x1_eval, x2_eval, t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)

    path_samples_train = '{}/recon_extra/{}/train_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                           checkpoint['current_epoch'],
                                                                           eval_args.seed, eval_args.t)
    path_samples_eval = '{}/recon_extra/{}/eval_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                         checkpoint['current_epoch'],
                                                                         eval_args.seed, eval_args.t)

    vutils.save_image(train_samples, fp=path_samples_train, nrow=18)
    vutils.save_image(eval_samples, fp=path_samples_eval, nrow=18)

# first extrapolate in X, and reconstruct all the points
elif eval_args.choice == 'extra_recon':
    path_samples = '{}/extra_recon/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                               eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    path_samples_time = '{}/extra_recon/{}/sample_ep{}_s{}.png'.format(eval_args.model, args.name,
                                                                       checkpoint['current_epoch'],
                                                                       eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples_time)):
        os.mkdir(os.path.dirname(path_samples_time))

    # Data Loader
    train_loader_extra_recon = DataLoader(dataset.train, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                          pin_memory=args.pin_memory)
    eval_loader_extra_recon = DataLoader(dataset.test, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                         pin_memory=args.pin_memory)

    train_i = 1
    x1_train = None
    x2_train = None
    for x in train_loader_extra_recon:
        x = x.to(device)
        if train_i == 1:
            x1_train = x
            train_i = train_i + 1
        elif train_i == 2:
            x2_train = x
            train_i = train_i + 1
        else:
            break

    eval_i = 1
    x1_eval = None
    x2_eval = None
    for y in eval_loader_extra_recon:
        y = y.to(device)
        if eval_i == 1:
            x1_eval = y
            eval_i = eval_i + 1
        elif eval_i == 2:
            x2_eval = y
            eval_i = eval_i + 1
        else:
            break

    train_list = []
    for i in np.arange(-1, 0, 0.125):
        x1_i = (1 - i) * x1_train + i * x2_train
        train_list.append(x1_i)
    train_list.append(x1_train)
    train_list.append(x2_train)
    for i in np.arange(1.125, 2.125, 0.125):
        x2_i = (1 - i) * x1_train + i * x2_train
        train_list.append(x2_i)
    x_train = torch.stack(train_list, dim=0)
    x_train = x_train.squeeze(dim=1)

    eval_list = []
    for i in np.arange(-1, 0, 0.125):
        x1_i = (1 - i) * x1_eval + i * x2_eval
        eval_list.append(x1_i)
    eval_list.append(x1_eval)
    eval_list.append(x2_eval)
    for i in np.arange(1.125, 2.125, 0.125):
        x2_i = (1 - i) * x1_eval + i * x2_eval
        eval_list.append(x2_i)
    x_eval = torch.stack(eval_list, dim=0)
    x_eval = x_eval.squeeze(dim=1)

    train_samples = model.forward_reconstruction(x_train, t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)
    eval_samples = model.forward_reconstruction(x_eval, t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)

    path_samples_train = '{}/extra_recon/{}/train_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                           checkpoint['current_epoch'],
                                                                           eval_args.seed, eval_args.t)
    path_samples_eval = '{}/extra_recon/{}/eval_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                         checkpoint['current_epoch'],
                                                                         eval_args.seed, eval_args.t)

    vutils.save_image(train_samples, fp=path_samples_train, nrow=18)
    vutils.save_image(eval_samples, fp=path_samples_eval, nrow=18)
