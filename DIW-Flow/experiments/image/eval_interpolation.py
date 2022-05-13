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
parser.add_argument('--choice', type=str, default='two_points_single',
                    choices=['two_points_single', 'two_points_multiple', 'two_classes', 'recon_inter', 'inter_recon',
                             'inter_recon_class', 'recon_inter_class',
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

# interpolate in latent space but only in the smallest scale
if eval_args.choice == 'two_points_single':

    path_samples = '{}/two_points_single/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                                     eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    samples = model.two_points_single(t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)
    path_samples_t = '{}/two_points_single/ep{}_s{}_t{}_{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                                       eval_args.seed, eval_args.t, args.name)
    vutils.save_image(samples, fp=path_samples_t, nrow=9)

# interpolate in the latent space with multi-scale
elif eval_args.choice == 'two_points_multiple':
    path_samples = '{}/two_points_multiple/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                                       eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    samples = model.two_points_multiple(t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)
    path_samples_t = '{}/two_points_multiple/ep{}_s{}_t{}_{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                                         eval_args.seed, eval_args.t, args.name)
    vutils.save_image(samples, fp=path_samples_t, nrow=9)

# sample a mini-batch in X and forward to Z, and backward to X
elif eval_args.choice == 'reconstruction':
    path_samples = '{}/reconstruction/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                                  eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    path_samples_time = '{}/reconstruction/{}/sample_ep{}_s{}.png'.format(eval_args.model, args.name,
                                                                          checkpoint['current_epoch'],
                                                                          eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples_time)):
        os.mkdir(os.path.dirname(path_samples_time))

    # Data Loader
    train_loader_reconstruct = DataLoader(dataset.train, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                          pin_memory=args.pin_memory)
    eval_loader_reconstruct = DataLoader(dataset.test, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                         pin_memory=args.pin_memory)

    train_list = []
    train_i = 1
    for x in train_loader_reconstruct:
        x = x.to(device)
        train_list.append(x)
        if train_i == eval_args.numbers:
            break
        else:
            train_i = train_i + 1
    train_x = torch.stack(train_list, dim=0)
    train_x = train_x.squeeze(dim=1)

    eval_list = []
    eval_i = 1
    for y in eval_loader_reconstruct:
        y = y.to(device)
        eval_list.append(y)
        if eval_i == eval_args.numbers:
            break
        else:
            eval_i = eval_i + 1
    eval_x = torch.stack(eval_list, dim=0)
    eval_x = eval_x.squeeze(dim=1)

    train_samples = model.forward_reconstruction(train_x, t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)
    eval_samples = model.forward_reconstruction(eval_x, t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)

    path_samples_train = '{}/reconstruction/{}/train_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                              checkpoint['current_epoch'],
                                                                              eval_args.seed, eval_args.t)
    path_samples_eval = '{}/reconstruction/{}/eval_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                            checkpoint['current_epoch'],
                                                                            eval_args.seed, eval_args.t)
    vutils.save_image(train_samples, fp=path_samples_train, nrow=eval_args.numbers)
    vutils.save_image(eval_samples, fp=path_samples_eval, nrow=eval_args.numbers)

# first interpolate in X, then forward to Z, and interpolate in Z with the sample interpolation method, and backward
# to X
elif eval_args.choice == 'recon_inter':
    path_samples = '{}/recon_inter/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                               eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    path_samples_time = '{}/recon_inter/{}/sample_ep{}_s{}.png'.format(eval_args.model, args.name,
                                                                       checkpoint['current_epoch'],
                                                                       eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples_time)):
        os.mkdir(os.path.dirname(path_samples_time))

    # Data Loader
    train_loader_recon_inter = DataLoader(dataset.train, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                          pin_memory=args.pin_memory)
    eval_loader_recon_inter = DataLoader(dataset.test, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                         pin_memory=args.pin_memory)

    train_i = 1
    x1_train = None
    x2_train = None
    for x in train_loader_recon_inter:
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
    for y in eval_loader_recon_inter:
        y = y.to(device)
        if eval_i == 1:
            x1_eval = y
            eval_i = eval_i + 1
        elif eval_i == 2:
            x2_eval = y
            eval_i = eval_i + 1
        else:
            break

    train_samples = model.forward_recon_inter(x1_train, x2_train, t=eval_args.t).cpu().float() / (
            2 ** args.num_bits - 1)
    eval_samples = model.forward_recon_inter(x1_eval, x2_eval, t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)

    path_samples_train = '{}/recon_inter/{}/train_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                           checkpoint['current_epoch'],
                                                                           eval_args.seed, eval_args.t)
    path_samples_eval = '{}/recon_inter/{}/eval_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                         checkpoint['current_epoch'],
                                                                         eval_args.seed, eval_args.t)

    vutils.save_image(train_samples, fp=path_samples_train, nrow=9)
    vutils.save_image(eval_samples, fp=path_samples_eval, nrow=9)

# do the same operation as 'recon_inter' but only interpolate between two samples of the same class
elif eval_args.choice == 'recon_inter_class':
    raise NotImplementedError('Still not finish yet')

# first interpolate in X, and reconstruct all the points
elif eval_args.choice == 'inter_recon':
    path_samples = '{}/inter_recon/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
                                                               eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    path_samples_time = '{}/inter_recon/{}/sample_ep{}_s{}.png'.format(eval_args.model, args.name,
                                                                       checkpoint['current_epoch'],
                                                                       eval_args.seed)
    if not os.path.exists(os.path.dirname(path_samples_time)):
        os.mkdir(os.path.dirname(path_samples_time))

    # Data Loader
    train_loader_inter_recon = DataLoader(dataset.train, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                          pin_memory=args.pin_memory)
    eval_loader_inter_recon = DataLoader(dataset.test, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                         pin_memory=args.pin_memory)

    train_i = 1
    x1_train = None
    x2_train = None
    for x in train_loader_inter_recon:
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
    for y in eval_loader_inter_recon:
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
    train_list.append(x1_train)
    for i in np.arange(0.125, 1, 0.125):
        x_i = (1 - i) * x1_train + i * x2_train
        train_list.append(x_i)
    train_list.append(x2_train)
    x_train = torch.stack(train_list, dim=0)
    x_train = x_train.squeeze(dim=1)

    eval_list = []
    eval_list.append(x1_eval)
    for i in np.arange(0.125, 1, 0.125):
        y_i = (1 - i) * x1_eval + i * x2_eval
        eval_list.append(y_i)
    eval_list.append(x2_eval)
    x_eval = torch.stack(eval_list, dim=0)
    x_eval = x_eval.squeeze(dim=1)

    train_samples = model.forward_reconstruction(x_train, t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)
    eval_samples = model.forward_reconstruction(x_eval, t=eval_args.t).cpu().float() / (2 ** args.num_bits - 1)

    path_samples_train = '{}/inter_recon/{}/train_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                           checkpoint['current_epoch'],
                                                                           eval_args.seed, eval_args.t)
    path_samples_eval = '{}/inter_recon/{}/eval_ep{}_s{}_t{}.png'.format(eval_args.model, args.name,
                                                                         checkpoint['current_epoch'],
                                                                         eval_args.seed, eval_args.t)

    vutils.save_image(train_samples, fp=path_samples_train, nrow=9)
    vutils.save_image(eval_samples, fp=path_samples_eval, nrow=9)

# do the same thing as 'inter_recon' but only in the same class
elif eval_args.choice == 'inter_recon_class':
    raise NotImplementedError('still not finish yet')