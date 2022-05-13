'''
python eval_fid.py --model PATH_TO_MODEL
'''

import os
from shutil import rmtree
import math
import torch
import pickle
import argparse
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from pytorch_fid.fid_score import calculate_fid_given_paths  # calculate the FID score
from pytorch_fid.inception import InceptionV3

# Data
from data.data import get_data, get_data_id, add_data_args, get_data_shape, get_augmentation
from denseflow.data.loaders.image import CIFAR10, ImageNet32, ImageNet64, SVHN, MNIST, CIFAR10Supervised, \
    SVHNSupervised, CIFAR100Supervised, CelebA

# Model
from model.model_flow import get_model, get_model_id, add_model_args
from denseflow.distributions import DataParallelDistribution

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)  # model path till 'DF_74_10'
parser.add_argument('--batch_size', type=int,
                    default=64)  # please be careful with the difference between 'batch_size' and 'batch-size'
parser.add_argument('--size', type=int, default=50000)  # total number of generated images
parser.add_argument('--seed', type=int, default=0)

# pytorch_fid setting
parser.add_argument('--batch_size_fid', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
'''
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))
'''

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)  # fix random number that uniquely decided by seed

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

##################
## Specify data ##
##################

train_loader, eval_loader, data_shape, dataset = get_data(args)

# also save training datasets and validation datasets in different places
'''
data_shape_fid = get_data_shape(args.dataset)
pil_transforms = get_augmentation(args.augmentation, args.dataset, data_shape_fid)
if args.dataset == 'cifar10':
    dataset = CIFAR10(num_bits=args.num_bits, pil_transforms=pil_transforms)
elif args.dataset == 'celeba':
    dataset = CelebA(num_bits=args.num_bits, pil_transforms=pil_transforms)
elif args.dataset == 'imagenet32':
    dataset = ImageNet32(num_bits=args.num_bits, pil_transforms=pil_transforms)
elif args.dataset == 'imagenet64':
    dataset = ImageNet64(num_bits=args.num_bits, pil_transforms=pil_transforms)
elif args.dataset == 'svhn':
    dataset = SVHN(num_bits=args.num_bits, pil_transforms=pil_transforms)
elif args.dataset == 'svhnsup':
    dataset = SVHNSupervised(num_bits=args.num_bits, pil_transforms=pil_transforms)
elif args.dataset == 'mnist':
    dataset = MNIST(num_bits=args.num_bits, pil_transforms=pil_transforms)
else:
    raise NotImplementedError()
'''

# Data Loader
train_loader_fid = DataLoader(dataset.train, batch_size=1, shuffle=True, num_workers=args.num_workers,
                          pin_memory=args.pin_memory)
valid_loader_fid = DataLoader(dataset.test, batch_size=1, shuffle=True, num_workers=args.num_workers,
                          pin_memory=args.pin_memory)


path_real = '{}/fid_real/real.txt'.format(eval_args.model)
if not os.path.exists(os.path.dirname(path_real)):
    os.mkdir(os.path.dirname(path_real))

# training dataset
path_real_train = '{}/fid_real/train/train.txt'.format(eval_args.model)
if not os.path.exists(os.path.dirname(path_real_train)):
    os.mkdir(os.path.dirname(path_real_train))

    #    train_loader = DataLoader(dataset.train, batch_size=1, shuffle=True, num_workers=args.num_workers,
    #                              pin_memory=False)
i = 0
for x in train_loader_fid:
    path_real_train = '{}/fid_real/train/train_{}.png'.format(eval_args.model, i + 1)
    x = x.float() / (2 ** args.num_bits - 1)
    vutils.save_image(x, fp=path_real_train, nrow=1)
    print('{}/{}'.format(i + 1, len(train_loader_fid)), end='\r')
    i = i + 1
path_real_train = '{}/fid_real/train'.format(eval_args.model)  # training dataset path
print('successfully split training dataset')

# validation dataset
path_real_valid = '{}/fid_real/valid/valid.txt'.format(eval_args.model)
if not os.path.exists(os.path.dirname(path_real_valid)):
    os.mkdir(os.path.dirname(path_real_valid))
    #    valid_loader = DataLoader(dataset.test, batch_size=1, shuffle=True, num_workers=args.num_workers,
    #                              pin_memory=False)
i = 0
for x in valid_loader_fid:
    path_real_valid = '{}/fid_real/valid/valid_{}.png'.format(eval_args.model, i + 1)
    x = x.float() / (2 ** args.num_bits - 1)
    vutils.save_image(x, fp=path_real_valid, nrow=1)
    print('{}/{}'.format(i + 1, len(valid_loader_fid)), end='\r')
    i = i + 1
path_real_valid = '{}/fid_real/valid'.format(eval_args.model)  # validation dataset path
print('successfully split validation dataset')

# Adjust args
args.batch_size = eval_args.batch_size

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
## FID ##
############

device = 'cuda:6' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()

# generate images
path_generate = '{}/fid_generate/generate.txt'.format(eval_args.model)
if not os.path.exists(os.path.dirname(path_generate)):
    os.mkdir(os.path.dirname(path_generate))
for n in range(1000):
    assert eval_args.size % 1000 == 0
    torch.cuda.empty_cache()  # the memory is insufficient
    samples = model.sample(int(eval_args.size/1000)).cpu().float() / (2 ** args.num_bits - 1)
    for i in range(int(eval_args.size/1000)):
        y = samples[i]
        path_generate = '{}/fid_generate/generate_{}.png'.format(eval_args.model, n * int(eval_args.size / 1000) + i + 1)
        vutils.save_image(y, fp=path_generate, nrow=1)
        print('epoch{} {}/{}'.format(n+1, i + 1, eval_args.size), end='\r')
print('successfully generate images')
path_generate = '{}/fid_generate'.format(eval_args.model)  # generated images path

# calculate the FID score corresponding to training and validation datasets respectively
# fid = dataset_fid(model, eval_loader, device=device)
if args.num_workers is None:
    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)
else:
    num_workers = args.num_workers

path_training = [path_generate, path_real_train]
path_validation = [path_generate, path_real_valid]

fid_training = calculate_fid_given_paths(path_training, eval_args.batch_size_fid, device, eval_args.dims, num_workers)
print('FID(training):', fid_training)
fid_validation = calculate_fid_given_paths(path_validation, eval_args.batch_size_fid, device, eval_args.dims,
                                           num_workers)
print('FID(validation):', fid_validation)

path_fid = '{}/fid/ep{}.txt'.format(eval_args.model, checkpoint['current_epoch'])
if not os.path.exists(os.path.dirname(path_fid)):
    os.mkdir(os.path.dirname(path_fid))

path_fid_training = '{}/fid/ep{}_size{}_training.txt'.format(eval_args.model, checkpoint['current_epoch'],
                                                             eval_args.size)
path_fid_validation = '{}/fid/ep{}_size{}_validation.txt'.format(eval_args.model, checkpoint['current_epoch'],
                                                                 eval_args.size)
with open(path_fid_training, 'w') as f:
    f.write(str(fid_training))
with open(path_fid_validation, 'w') as f:
    f.write(str(fid_validation))

# delete the image file
print('delete the training dataset')
rmtree(path_real_train)
print('delete the validation dataset')
rmtree(path_real_valid)
print('delete the generated images')
rmtree(path_generate)
# remember to empty the Recycle Bin !!!
# sudo rm -rf ~/.local/share/Trash/*
