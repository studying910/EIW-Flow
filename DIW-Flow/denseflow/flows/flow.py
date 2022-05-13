import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from collections.abc import Iterable
from denseflow.distributions import Distribution
from denseflow.transforms import Transform
from experiments.image.model.flow_modules import InvertibleDenseBlock, InvertibleTransition
import experiments.image.model as model


class Flow(Distribution):
    """
    Base class for Flow.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    def __init__(self, base_dist, transforms, coef=1.):
        super(Flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)
        self.coef = coef


    def log_prob(self, x, return_z=False):

        log_prob = torch.zeros(x.shape[0], device=x.device)
        loss_list = []
        for transform in self.transforms:
            if transform.__class__.__name__ == 'InvertibleDenseBlock':  # only this way can get the name:
                x, ldj = transform(x)
                log_prob += ldj
            elif transform.__class__.__name__ == 'InvertibleTransition':
                x, ldj, loss = transform(x)
                loss_list.append(loss)
                log_prob += ldj
            else:
                x, ldj = transform(x)
                log_prob += ldj
        # log_prob = log_prob / self.base_dist.scale
        log_prob += self.base_dist.log_prob(x)
        log_prob = log_prob / self.coef
        scale_loss = torch.stack(loss_list, dim=0)

        if return_z:
            return x, log_prob
        return log_prob, scale_loss.sum()
    
    def forward_reconstruction(self, x, t=1.0):
        x_real = x
        for transform in self.transforms:
            if transform.__class__.__name__ == 'InvertibleDenseBlock':  # only this way can get the name
                x, _ = transform(x)
            elif transform.__class__.__name__ == 'InvertibleTransition':
                x, _, _ = transform(x)
            else:
                x, _ = transform(x)

        z = x * t
        for transform in reversed(self.transforms):
            z = transform.inverse(z)

        z_final = torch.cat((x_real, z), dim=0)
        return z_final
    
    def forward_recon_inter(self, x1, x2, t=1.0):
        x_list = []
        x_list.append(x1)
        for i in np.arange(0.125, 1, 0.125):
            x_i = (1 - i) * x1 + i * x2
            x_list.append(x_i)
        x_list.append(x2)
        x = torch.stack(x_list, dim=0)
        x = x.squeeze(dim=1)

        for transform in self.transforms:
            if transform.__class__.__name__ == 'InvertibleDenseBlock':  # only this way can get the name
                x1, _ = transform(x1)
                x2, _ = transform(x2)
            elif transform.__class__.__name__ == 'InvertibleTransition':
                x1, _, _ = transform(x1)
                x2, _, _ = transform(x2)
            else:
                x1, _ = transform(x1)
                x2, _ = transform(x2)

        z1 = x1 * t
        z2 = x2 * t
        z_list = []
        for i in np.arange(0.125, 1, 0.125):
            z_i = (1 - i) * z1 + i * z2
            z_list.append(z_i)
        for transform in reversed(self.transforms):
            if transform.__class__.__name__ == 'InvertibleTransition':
                z1, z12 = transform.inverse_points(z1)
                z2, z22 = transform.inverse_points(z2)
                z_list_new = []
                i = 0.125
                for z_i in z_list:
                    z_i2 = (1 - i) * z12 + i * z22
                    z_i = transform.inverse_inter(z_i, z_i2)
                    z_list_new.append(z_i)
                    i = i + 0.125
                z_list = z_list_new
            else:
                z1 = transform.inverse(z1)
                z2 = transform.inverse(z2)
                z_list_new = []
                for z_i in z_list:
                    z_i = transform.inverse(z_i)
                    z_list_new.append(z_i)
                z_list = z_list_new

        z_list_final = []
        z_list_final.append(z1)
        for z_i in z_list:
            z_list_final.append(z_i)
        z_list_final.append(z2)
        z = torch.stack(z_list_final, dim=0)
        z = z.squeeze(dim=1)

        z_final = torch.cat((x, z), dim=0)
        return z_final
    
    def forward_recon_extra(self, x1, x2, t=1.0):
        x_list = []
        for i in np.arange(-1, 0, 0.125):
            x1_i = (1 - i) * x1 + i * x2
            x_list.append(x1_i)
        x_list.append(x1)
        x_list.append(x2)
        for i in np.arange(1.125, 2.125, 0.125):
            x2_i = (1 - i) * x1 + i * x2
            x_list.append(x2_i)
        x = torch.stack(x_list, dim=0)
        x = x.squeeze(dim=1)

        for transform in self.transforms:
            if transform.__class__.__name__ == 'InvertibleDenseBlock':  # only this way can get the name
                x1, _ = transform(x1)
                x2, _ = transform(x2)
            elif transform.__class__.__name__ == 'InvertibleTransition':
                x1, _, _ = transform(x1)
                x2, _, _ = transform(x2)
            else:
                x1, _ = transform(x1)
                x2, _ = transform(x2)

        z1 = x1 * t
        z2 = x2 * t
        z1_list = []
        for i in np.arange(-0.875, 0, 0.125):
            z1_i = (1 - i) * z1 + i * z2
            z1_list.append(z1_i)
        z1_left = 2 * z1 - z2

        z2_list = []
        for i in np.arange(1.125, 2, 0.125):
            z2_i = (1 - i) * z1 + i * z2
            z2_list.append(z2_i)
        z2_right = -z1 + 2 * z2

        for transform in reversed(self.transforms):
            if transform.__class__.__name__ == 'InvertibleTransition':
                z1, z12 = transform.inverse_points(z1)
                z2, z22 = transform.inverse_points(z2)
                z1_left, z12_left = transform.inverse_points(z1_left)
                z2_right, z22_right = transform.inverse_points(z2_right)

                z1_list_new = []
                i1 = 0.125
                for z1_i in z1_list:
                    z1_i2 = (1 - i1) * z12_left + i1 * z12
                    z1_i = transform.inverse_inter(z1_i, z1_i2)
                    z1_list_new.append(z1_i)
                    i1 = i1 + 0.125
                z1_list = z1_list_new

                z2_list_new = []
                i2 = 0.125
                for z2_i in z2_list:
                    z2_i2 = (1 - i2) * z22 + i2 * z22_right
                    z2_i = transform.inverse_inter(z2_i, z2_i2)
                    z2_list_new.append(z2_i)
                    i2 = i2 + 0.125
                z2_list = z2_list_new
            else:
                z1 = transform.inverse(z1)
                z2 = transform.inverse(z2)
                z1_left = transform.inverse(z1_left)
                z2_right = transform.inverse(z2_right)

                z1_list_new = []
                for z1_i in z1_list:
                    z1_i = transform.inverse(z1_i)
                    z1_list_new.append(z1_i)
                z1_list = z1_list_new
                z2_list_new = []
                for z2_i in z2_list:
                    z2_i = transform.inverse(z2_i)
                    z2_list_new.append(z2_i)
                z2_list = z2_list_new

        z_list_final = []
        z_list_final.append(z1_left)
        for z1_i in z1_list:
            z_list_final.append(z1_i)
        z_list_final.append(z1)
        z_list_final.append(z2)
        for z2_i in z2_list:
            z_list_final.append(z2_i)
        z_list_final.append(z2_right)
        z = torch.stack(z_list_final, dim=0)
        z = z.squeeze(dim=1)

        z_final = torch.cat((x, z), dim=0)
        return z_final

    def sample(self, num_samples, t=1.0):
        z = self.base_dist.sample(num_samples)
        z = z * t
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z
    
    def two_points_single(self, t=1.0):
        z1 = self.base_dist.sample(1)
        z2 = self.base_dist.sample(1)
        z1 = z1 * t  # what will happen if we interpolate between two variables with different temperature
        z2 = z2 * t
        z_list = []
        z_list.append(z1)
        for i in np.arange(0.0125, 1, 0.0125):
            z_i = (1 - i) * z1 + i * z2
            z_list.append(z_i)
        z_list.append(z2)
        z = torch.stack(z_list, dim=0)
        z = z.squeeze(dim=1)
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z
    
    def two_points_single_extra(self, t=1.0):
        z1 = self.base_dist.sample(1)
        z2 = self.base_dist.sample(1)
        z1 = z1 * t
        z2 = z2 * t
        z_list = []
        for i in np.arange(-1, 0, 0.125):
            z_i = (1 - i) * z1 + i * z2
            z_list.append(z_i)
        z_list.append(z1)
        z_list.append(z2)
        for i in np.arange(1.125, 2.125, 0.125):
            z_i = (1 - i) * z1 + i * z2
            z_list.append(z_i)
        z = torch.stack(z_list, dim=0)
        z = z.squeeze(dim=1)
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

    def two_points_multiple(self, t=1.0):
        z1 = self.base_dist.sample(1)
        z2 = self.base_dist.sample(1)
        z1 = z1 * t
        z2 = z2 * t
        z_list = []
        for i in np.arange(0.0588, 0.94996, 0.0588):
            z_i = (1 - i) * z1 + i * z2
            z_list.append(z_i)
        for transform in reversed(self.transforms):
            if transform.__class__.__name__ == 'InvertibleTransition':
                z1, z12 = transform.inverse_points(z1)
                z2, z22 = transform.inverse_points(z2)
                z_list_new = []
                i = 0.0588
                for z_i in z_list:
                    z_i2 = (1 - i) * z12 + i * z22
                    z_i = transform.inverse_inter(z_i, z_i2)
                    z_list_new.append(z_i)
                    i = i + 0.0588
                z_list = z_list_new
            else:
                z1 = transform.inverse(z1)
                z2 = transform.inverse(z2)
                z_list_new = []
                for z_i in z_list:
                    z_i = transform.inverse(z_i)
                    z_list_new.append(z_i)
                z_list = z_list_new

        x_list = []
        x_list.append(z1)
        for z_i in z_list:
            x_list.append(z_i)
        x_list.append(z2)
        x = torch.stack(x_list, dim=0)
        x = x.squeeze(dim=1)
        return x
    
    def two_points_multiple_extra(self, t=1.0):
        z1 = self.base_dist.sample(1)
        z2 = self.base_dist.sample(1)
        z1 = z1 * t
        z2 = z2 * t

        z1_list = []
        for i in np.arange(-0.875, 0, 0.125):
            z1_i = (1 - i) * z1 + i * z2
            z1_list.append(z1_i)
        z1_left = 2 * z1 - z2

        z2_list = []
        for i in np.arange(1.125, 2, 0.125):
            z2_i = (1 - i) * z1 + i * z2
            z2_list.append(z2_i)
        z2_right = -z1 + 2 * z2

        for transform in reversed(self.transforms):
            if transform.__class__.__name__ == 'InvertibleTransition':
                z1, z12 = transform.inverse_points(z1)
                z2, z22 = transform.inverse_points(z2)
                z1_left, z12_left = transform.inverse_points(z1_left)
                z2_right, z22_right = transform.inverse_points(z2_right)

                z1_list_new = []
                i1 = 0.125
                for z1_i in z1_list:
                    z1_i2 = (1 - i1) * z12_left + i1 * z12
                    z1_i = transform.inverse_inter(z1_i, z1_i2)
                    z1_list_new.append(z1_i)
                    i1 = i1 + 0.125
                z1_list = z1_list_new

                z2_list_new = []
                i2 = 0.125
                for z2_i in z2_list:
                    z2_i2 = (1 - i2) * z22 + i2 * z22_right
                    z2_i = transform.inverse_inter(z2_i, z2_i2)
                    z2_list_new.append(z2_i)
                    i2 = i2 + 0.125
                z2_list = z2_list_new
            else:
                z1 = transform.inverse(z1)
                z2 = transform.inverse(z2)
                z1_left = transform.inverse(z1_left)
                z2_right = transform.inverse(z2_right)

                z1_list_new = []
                for z1_i in z1_list:
                    z1_i = transform.inverse(z1_i)
                    z1_list_new.append(z1_i)
                z1_list = z1_list_new
                z2_list_new = []
                for z2_i in z2_list:
                    z2_i = transform.inverse(z2_i)
                    z2_list_new.append(z2_i)
                z2_list = z2_list_new

        x_list = []
        x_list.append(z1_left)
        for z1_i in z1_list:
            x_list.append(z1_i)
        x_list.append(z1)
        x_list.append(z2)
        for z2_i in z2_list:
            x_list.append(z2_i)
        x_list.append(z2_right)
        x = torch.stack(x_list, dim=0)
        x = x.squeeze(dim=1)
        return x
    
    def forward_show_middle_status(self, x, channel_nums):
        # test shape of x : [1,3,32,32]

        middle_status = []

        for transform in self.transforms:
            if transform.__class__.__name__ == 'InvertibleTransition':
                after_sort, before_sort, _ = transform.get_before_after_sort(x)
                # show part of the feature maps
                before_sort_head = torch.unsqueeze(torch.squeeze(before_sort[:, :channel_nums, :, :]), dim=1)
                before_sort_tail = torch.unsqueeze(torch.squeeze(before_sort[:, -channel_nums:, :, :]), dim=1)

                after_sort_head = torch.unsqueeze(torch.squeeze(after_sort[:, :channel_nums, :, :]), dim=1)
                after_sort_tail = torch.unsqueeze(torch.squeeze(after_sort[:, -channel_nums:, :, :]), dim=1)

                middle_status.append(torch.cat([before_sort_head, before_sort_tail, after_sort_head, after_sort_tail], dim=0))
                x, _, _ = transform(x)
            else:
                x, _ = transform(x)

        return middle_status
    
    def forward_keep_status(self, x):
        
        for transform in self.transforms:
            if transform.__class__.__name__ == 'InvertibleTransition':
                x, _, _ = transform.forward_keep_status(x)

            else:
                x, _ = transform(x)
                
        return x
    
    def inverse_keep_status(self, x, keep=[]):
        # keep: choose which scale should be keep, start from 1, max(keep) <= args.block_conf - 1
        scale_num = 1

        # forward
        for transform in self.transforms:
            if transform.__class__.__name__ == 'InvertibleTransition':
                if scale_num in keep:
                    x, _, _ = transform.forward_keep_status(x)
                else:
                    x, _, _ = transform(x)

                scale_num = scale_num + 1
                print(scale_num)

            else:
                x, _ = transform(x)

        # inverse
        scale_num = scale_num - 1
        for transform in reversed(self.transforms):
            if transform.__class__.__name__ == 'InvertibleTransition':
                if scale_num in keep:
                    x = transform.inverse_keep_status(x)
                else:
                    x = transform.inverse(x)
                scale_num = scale_num - 1
                print(scale_num)
                print('No')
            else:
                x = transform.inverse(x)
        return x
    
    def cal_KLDivloss(self, x, scale_num):
        # x.shape: B x C x H x W
        # F.kl_div(),   scale_num : CelebA:4, others:3
#         part_list = torch.Tensor([0]*4).to(x)
        part_list = [0] * 4
        # forward
        KL_matrix_batch = torch.zeros((x.shape[0], scale_num-1, 4)).to(x)
        scale = 0
        for transform in self.transforms:
            if transform.__class__.__name__ == 'InvertibleTransition':
                x_before_sort, x_after_sort, _ = transform.get_before_after_sort(x)
                # before
                part_list[0], part_list[1] = torch.tensor_split(x_before_sort, 2, dim=1)   # split across channel dim
                # after
                part_list[2], part_list[3] = torch.tensor_split(x_after_sort, 2, dim=1)

                ##### cal KLD
#                 print(part_list[1])
                for i in range(4):
                    # result: [B, C/2, H, W]  
                    print(part_list[i])
                    KL_batch = F.kl_div(torch.log(part_list[i]), torch.randn_like(part_list[i]).to(x), reduction='none')

                    # reshape and calculate the mean of each image of this batch
                    KL_matrix_batch[:, scale, i] = torch.mean(KL_batch.view(KL_batch.shape[0], -1), dim=1)    # shape: [batch_size, 1]

                # sample
                scale = scale + 1
#                 print(KL_batch)
                
                x, _, _ = transform(x)

            else:
                x, _ = transform(x)
        return KL_matrix_batch

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")
