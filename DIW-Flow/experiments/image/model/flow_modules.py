import torch
import torch.nn as nn
from denseflow.transforms import Transform, Squeeze2d, Slice, WaveletSqueeze2d, SwitchChannels
from denseflow.transforms import Conv1x1, BatchNormBijection2d, ActNormBijection2d
from denseflow.distributions import StandardNormal
from denseflow.nn.layers import ElementwiseParams2d
from denseflow.nn.nets import DenseNet
from denseflow.utils import sum_except_batch
import torch.nn.functional as F

from .affine_coupling import SingleAffineCoupling
from .SENet import SELayer

class SequentialTransform(Transform):
    def __init__(self, transforms):
        super(SequentialTransform, self).__init__()
        assert type(transforms) == torch.nn.ModuleList
        self.transforms = transforms

    def forward(self, x):
        total_ld = torch.zeros(x.shape[0]).to(x)
        for layer in self.transforms:
            x, ld = layer(x)
            total_ld += ld
        return x, total_ld

    def inverse(self, z):
        for layer in reversed(self.transforms):
            z = layer.inverse(z)
        return z

class _BNReLUConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_BNReLUConv, self).__init__()
        self.transformation = nn.Sequential(
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4 * in_channels, 2 * out_channels, kernel_size=1))
        nn.init.zeros_(self.transformation[-1].weight)
        nn.init.zeros_(self.transformation[-1].bias)

    def forward(self, x):
        mu, unconstrained_scale = self.transformation(x).chunk(2, dim=1)
        log_scale = 2. * torch.tanh(unconstrained_scale / 2.)
        return mu, log_scale


class WeightGenerator(Transform):
    
    def __init__(self, in_channel):
        super(WeightGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, in_channel//2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel//2, in_channel, bias=False),
            nn.Softmax(dim=1))
        
        self.in_channel = self.type_convert(in_channel)
    
    @staticmethod
    def type_convert(data):
        return torch.tensor(data, dtype=torch.float32).view(1,1)
        
    def sort(self):
        weight = self.fc(self.in_channel)
        _, indices = torch.sort(weight, descending=True)
        return weight, indices[0]
        
    def forward(self, in_channel, device=None):
        self.in_channel = self.in_channel.to(device)
        assert in_channel == self.in_channel
        return self.sort()
    
    def inverse(self, in_channel, device=None):
        self.in_channel = self.in_channel.to(device)
        assert self.type_convert(in_channel).to(device) == self.in_channel
        indices = self.sort()[1]
        _, indices_of_indices = torch.sort(indices)
        return indices_of_indices


class InvertibleTransition(Transform):

    def __init__(self, in_channels):
        super(InvertibleTransition, self).__init__()
        self.squeeze = Squeeze2d()
        self.dist = StandardNormal((in_channels * 2, 2, 2))
        self.SELayer = SELayer(channel=in_channels*4)    # we need to multiple channel by 4 for squeeze operation
        self.weightGenerator = WeightGenerator(in_channels*4)
        self.loss_cal = torch.nn.KLDivLoss(reduction='batchmean')
        self.loss = None
        
    def get_sort_loss(self):
        return self.loss

    def get_before_after_sort(self, x):
        # return image features after(x_) and before(x) sorting.
        x, _ = self.squeeze(x)
        weight, indices = self.weightGenerator(x.size()[1], device=x.device)

        x_ = x[:, indices, :, :]  # sort channel dimension using weights
        return x_, x, weight

    def forward(self, x):
        x, _, weight = self.get_before_after_sort(x )
        # x, _ = self.squeeze(x)
        # weight, indices = self.weightGenerator(x.size()[1], device=x.device)

        weight_SE = self.SELayer(x)
        weight = weight.view(1, -1).expand_as(weight_SE)
        loss = self.loss_cal(weight_SE, weight)

        # x = x[:, indices, :, :]   # sort channel dimension using weights
        x1, x2 = torch.chunk(x, 2, dim=1)
        logpz = self.dist.log_prob(x2)
        return x1, logpz, loss
    
    def forward_keep_status(self, x):
        _, x, _ = self.get_before_after_sort(x)

        loss = 0
        x1, x2 = torch.chunk(x, 2, dim=1)
        logpz = self.dist.log_prob(x2)
        return x1, logpz, loss

    def inverse(self, z1):
        z2 = self.dist.sample(z1.shape)
        z = torch.cat([z1, z2], dim=1)
        # recover
        indices_of_indices = self.weightGenerator.inverse(z.size()[1], device=z.device)
        z = z[:, indices_of_indices, :, :]
        
        z = self.squeeze.inverse(z)
        return z
    
    def inverse_points(self, z1):  # use for end points
        z2 = self.dist.sample(z1.shape)
        z = torch.cat([z1, z2], dim=1)
        # recover the distribution
        indices_of_indices = self.weightGenerator.inverse(z.size()[1], device=z.device)
        z = z[:, indices_of_indices, :, :]
        z = self.squeeze.inverse(z)
        return z, z2

    def inverse_inter(self, z1, z2):  # use for middle points
        z = torch.cat([z1, z2], dim=1)
        # recover the distribution
        indices_of_indices = self.weightGenerator.inverse(z.size()[1], device=z.device)
        z = z[:, indices_of_indices, :, :]
        z = self.squeeze.inverse(z)
        return z

    def inverse_keep_status(self, z1):
        z2 = self.dist.sample(z1.shape)
        z = torch.cat([z1, z2], dim=1)
        z = self.squeeze.inverse(z)
        return z


class InvertibleDenseBlock(Transform):

    def __init__(self, in_channels, num_steps, num_layers, layer_mid_chnls, growth_rate=None, checkpointing=False):
        super(InvertibleDenseBlock, self).__init__()

        self.num_steps = num_steps
        self.in_channels = in_channels
        self.growth_rate = growth_rate

        self.noise_params = nn.ModuleList([])

        units = []
        noise_in_chnls = self.compute_noise_inputs(in_channels, growth_rate, num_steps)
        for i in range(self.num_steps):
            chnls = int(in_channels + i * growth_rate)
            if i != self.num_steps - 1:
                self.noise_params.append(_BNReLUConv(noise_in_chnls[i], self.growth_rate))

            modules = [
                SequentialTransform(nn.ModuleList([
                    ActNormBijection2d(chnls),
                    Conv1x1(chnls),
                    # Intra-unit coupling
                    SingleAffineCoupling(chnls, mid_chnls=layer_mid_chnls,checkpointing=checkpointing),
                    SwitchChannels()
                ])) for _ in range(num_layers)]
            units.append(SequentialTransform(nn.ModuleList(modules)))
        self.blocks = nn.Sequential(*units)
        self.noise_dist = torch.distributions.Normal(0, 1)

    def compute_noise_inputs(self, in_c, gr, steps):
        inputs = []
        out = []
        for i in range(steps):
            if i != 0:
                inputs.append(in_c)
                in_c += gr
            else:
                inputs.append(in_c)
            out.append(sum(inputs))
        return out

    def cross_unit_coupling(self, x, g):
        # Cross-unit coupling
        N, _, H, W = x.shape
        eta = self.noise_dist.sample(torch.Size([N, self.growth_rate, H, W])).to(x)
        mu, log_scale = g(x)
        eta_hat = mu + torch.exp(log_scale) * eta
        log_p = self.noise_dist.log_prob(eta)
        return eta_hat, - log_p.sum(dim=(1, 2, 3)) + sum_except_batch(log_scale)


    def forward(self, x):
        total_ld = torch.zeros(x.shape[0]).to(x)
        N, _, H, W = x.shape
        g_inputs = []
        x_in = x
        for i, block in enumerate(self.blocks.children()):
            if i != 0:
                g_in = torch.cat(g_inputs, dim=1)
                eta, delta_ll = self.cross_unit_coupling(g_in, self.noise_params[i-1])
                g_inputs.append(x_in)
                x_in = torch.cat((x_in, eta), dim=1)
                x_out, ldi = block(x_in)
                x_in = x_out
                ldi += delta_ll
            else:
                g_inputs.append(x_in)
                x_out, ldi = block(x_in)
                x_in = x_out

            total_ld = total_ld + ldi
        return x_out, total_ld

    def inverse(self, z):
        for i, block in enumerate(reversed(list(self.blocks.children()))):

            z = block.inverse(z)
            if i != self.num_steps - 1:
                z = z[:, :-self.growth_rate]
        return z
