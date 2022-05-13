import argparse
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np

import pickle
import os
import time
from tqdm import tqdm

# Data
from data.data import get_data, get_data_id, add_data_args

# Model
from model.model_flow import get_model, get_model_id, add_model_args
from denseflow.distributions import DataParallelDistribution

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--double', type=eval, default=False)
parser.add_argument('--numbers', type=int, default=9)
parser.add_argument('--channel_nums', type=int, default=4)
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)  # the same seed produces the same result

# def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
#     """
#     将tensor保存为pillow
#     :param input_tensor: 要保存的tensor
#     :param filename: 保存的文件名
#     """
#     assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
#     # 复制一份
#     input_tensor = input_tensor.clone().detach()
#     # 到cpu
#     input_tensor = input_tensor.to(torch.device('cpu'))
#     # 反归一化
#     # input_tensor = unnormalize(input_tensor)
#     # 去掉批次维度
#     input_tensor = input_tensor.squeeze()
#     # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
#     input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
#     # 转成pillow
#     im = Image.fromarray(input_tensor)
#     im.save(filename)

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

# Data Loader
train_loader = DataLoader(dataset.train, batch_size=16, shuffle=True, num_workers=args.num_workers,
                          pin_memory=args.pin_memory)
eval_loader = DataLoader(dataset.test, batch_size=16, shuffle=True, num_workers=args.num_workers,
                         pin_memory=args.pin_memory)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)  # return an initialized model with args
if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check)
model.load_state_dict(checkpoint['model'])
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()
if eval_args.double: model = model.double()

###################
## Middle Status ##
###################

# train_list = []
# train_i = 0
# for x in train_loader:
#     x = x.to(device)
#     if train_i == eval_args.numbers:
#         break
#     else:
#         train_i = train_i + 1
#     train_list.append(x)
#     train_x = torch.stack(train_list, dim=0)
#     train_x = train_x.squeeze(dim=1)

# eval_list = []
# eval_i = 0
# for y in eval_loader:
#     y = y.to(device)
#
#     if eval_i == eval_args.numbers:
#         break
#     else:
#         eval_i = eval_i + 1
#     eval_list.append(y)
# eval_x = torch.stack(eval_list, dim=0)
# eval_x = eval_x.squeeze(dim=1)

########### get the middle status, here sample batchsize=1!
# path_samples = '{}/middle_status/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
#                                                              eval_args.seed)
# if not os.path.exists(os.path.dirname(path_samples)):
#     os.mkdir(os.path.dirname(path_samples))

# vutils.save_image(train_x.squeeze(), fp=path_samples)

# middle_status = model.forward_show_middle_status(train_x, eval_args.channel_nums)
#
# # print(train_x.squeeze().shape)
# for i, middle_status_i in enumerate(middle_status):
#     path_middle_status_i = '{}/middle_status/middle_status_{}_ep{}_s{}.png'.format(eval_args.model, i,
#                                                                                    checkpoint['current_epoch'],
#                                                                                    eval_args.seed)
#     if not os.path.exists(os.path.dirname(path_middle_status_i)):
#         os.mkdir(os.path.dirname(path_middle_status_i))
#     vutils.save_image(middle_status_i, fp=path_middle_status_i, nrow=2 * eval_args.channel_nums)

########## keep some middle status, and reverse to get the reconstruct fig
# path_keep_status = '{}/keep_status/keep_status_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'],
#                                                                     eval_args.seed)
# if not os.path.exists(os.path.dirname(path_keep_status)):
#     os.mkdir(os.path.dirname(path_keep_status))

# keep_status_reverse = model.inverse_keep_status(train_x, keep=[])
# print(torch.max(keep_status_reverse), torch.min(keep_status_reverse))
# vutils.save_image(keep_status_reverse.float(), fp=path_keep_status, nrow=9)


# ######## Calculate the KL div
# KL_matrix = [0] * len(train_loader)
# # KL_matrix = [0] * 10
# for k, x in tqdm(enumerate(train_loader)):
#     KL_matrix[k] = model.cal_KLDivloss(x.to(device), scale_num=3)
# #     print(KL_matrix[k])
#     break

# KL_matrix_cat = torch.cat(KL_matrix,dim=0)
# print(KL_matrix_cat.shape)
# # torch.save(KL_matrix_cat, '{}/KL_matrix_cat_imagenet32.pt'.format(eval_args.model))

###### execute time 
save_path_time= '{}/time_list'.format(eval_args.model)
if not os.path.exists(os.path.dirname(save_path_time)):
    os.mkdir(os.path.dirname(save_path_time))

time_list_rep = [{'sort':[0]*len(train_loader), 'keep_status':[0]*len(train_loader)}, {'sort':[0]*len(eval_loader), 'keep_status':[0]*len(eval_loader)}] 
for k, x in tqdm(enumerate(train_loader)):
#     if k > 2:
#         break

    begin_time = time.time()
    _, _ = model.log_prob(x.to(device))
    end_time = time.time()
    time_list_rep[0]['sort'][k] = end_time - begin_time

    begin_time = time.time()
    _ = model.forward_keep_status(x.to(device))
    end_time = time.time()
    time_list_rep[0]['keep_status'][k] = end_time - begin_time
    
for k, x in tqdm(enumerate(eval_loader)):
#     if k > 2:
#         break
    begin_time = time.time()
    _, _ = model.log_prob(x.to(device))
    end_time = time.time()
    time_list_rep[1]['sort'][k] = end_time - begin_time

    begin_time = time.time()
    _ = model.forward_keep_status(x.to(device))
    end_time = time.time()
    time_list_rep[1]['keep_status'][k] = end_time - begin_time

torch.save(time_list_rep, '{}/time_list_rep_celebA.pt'.format(save_path_time))

# time_list = [0]*len(train_loader)
# for k, x in tqdm(enumerate(train_loader)):
#     begin_time = time.time()
#     _, _ = model.log_prob(x.to(device))
#     end_time = time.time()
#     time_list[k] = end_time - begin_time
# torch.save(torch.Tensor(time_list), '{}/time_list_cifar10.pt'.format(eval_args.model))

# time_list_keep_status = [0]*len(train_loader)
# for k, x in tqdm(enumerate(train_loader)):
#     begin_time = time.time()
#     _ = model.forward_keep_status(x.to(device))
#     end_time = time.time()
#     time_list_keep_status[k] = end_time - begin_time
# torch.save(torch.Tensor(time_list_keep_status), '{}/time_list_keep_status_cifar10.pt'.format(eval_args.model))

