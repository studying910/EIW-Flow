import math
import torch
import numpy as np

sigma = 1e-3  # default hyper-parameter of scale_loss


# remember to output the scale_loss


def loglik_nats(model, x):
    """Compute the log-likelihood in nats."""
    return - model.log_prob(x)[0].mean(), model.log_prob(x)[1]  # calculate the scale_loss


def loglik_bpd(model, x):
    """Compute the log-likelihood in bits per dim."""
    return - model.log_prob(x)[0].sum() / (math.log(2) * x.shape.numel()), model.log_prob(x)[1]


def elbo_nats(model, x):
    """
    Compute the ELBO in nats.
    Same as .loglik_nats(), but may improve readability.
    """
    return loglik_nats(model, x)


def elbo_bpd(model, x):
    """
    Compute the ELBO in bits per dim.
    Same as .loglik_bpd(), but may improve readability.
    """
    return loglik_bpd(model, x)


def iwbo(model, x, k):
    x_stack = torch.cat([x for _ in range(k)], dim=0)
    ll_stack = model.log_prob(x_stack)[0]
    scale_loss = model.log_prob(x_stack)[1]
    ll = torch.stack(torch.chunk(ll_stack, k, dim=0))
    return torch.logsumexp(ll, dim=0) - math.log(k), scale_loss


def iwbo_batched(model, x, k, kbs):
    assert k % kbs == 0
    num_passes = k // kbs
    ll_batched = []
    scale_loss_list = []
    for i in range(num_passes):
        x_stack = torch.cat([x for _ in range(kbs)], dim=0)
        ll_stack = model.log_prob(x_stack)[0]
        scale_loss = model.log_prob(x_stack)[1]
        scale_loss_list.append(scale_loss)
        ll_batched.append(torch.stack(torch.chunk(ll_stack, kbs, dim=0)))
    ll = torch.cat(ll_batched, dim=0)
    scale_loss_list = [sl.tolist() for sl in scale_loss_list]
    ss = torch.tensor(scale_loss_list, dtype=torch.float32)
    return torch.logsumexp(ll, dim=0) - math.log(k), ss.mean()


def iwbo_nats(model, x, k, kbs=None):
    """Compute the IWBO in nats."""
    if kbs:
        return - iwbo_batched(model, x, k, kbs)[0].mean(), iwbo_batched(model, x, k, kbs)[1]
    else:
        return - iwbo(model, x, k)[0].mean(), iwbo(model, x, k)[1]


def iwbo_bpd(model, x, k, kbs=None):
    """Compute the IWBO in bits per dim."""
    if kbs:
        return - iwbo_batched(model, x, k, kbs)[0].sum() / (x.numel() * math.log(2)), iwbo_batched(model, x, k, kbs)[1]
    else:
        return - iwbo(model, x, k)[0].sum() / (x.numel() * math.log(2)), iwbo(model, x, k)[1]


def dataset_elbo_nats(model, data_loader, device, double=False, verbose=True):
    with torch.no_grad():
        nats = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            nats += elbo_nats(model, x)[0].cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i + 1, len(data_loader)), nats / count, end='\r')
    return nats / count


def dataset_elbo_bpd(model, data_loader, device, double=False, verbose=True):
    with torch.no_grad():
        bpd = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            bpd += elbo_bpd(model, x)[0].cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i + 1, len(data_loader)), bpd / count, end='\r')
    return bpd / count


def dataset_iwbo_nats(model, data_loader, k, device, double=False, kbs=None, verbose=True):
    with torch.no_grad():
        nats = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            nats += iwbo_nats(model, x, k=k, kbs=kbs)[0].cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i + 1, len(data_loader)), nats / count, end='\r')
    return nats / count


def dataset_iwbo_bpd(model, data_loader, k, device, double=False, kbs=None, verbose=True):
    with torch.no_grad():
        bpd = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            bpd += iwbo_bpd(model, x, k=k, kbs=kbs)[0].cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i + 1, len(data_loader)), bpd / count, end='\r')
    return bpd / count


def dataset_fid(model, data_loader, device, verbose=True):
    # please use eval_fid.py to calculate the FID score
    raise NotImplementedError()


def mc_bpd_batched(model, x, k, kbs):
    assert k % kbs == 0
    num_passes = k // kbs
    bpd = 0.
    count = 0
    scale_loss_list = []
    for i in range(num_passes):
        x_stack = torch.cat([x for _ in range(kbs)], dim=0)
        bpd += elbo_bpd(model, x_stack)[0].cpu().item() * len(x_stack)
        scale_loss = elbo_bpd(model, x_stack)[1]
        scale_loss_list.append(scale_loss)
        count += len(x_stack)
    ss = torch.cat(scale_loss_list, dim=0)
    return bpd / count, ss.mean()


def dataset_mc_bpd(model, data_loader, k, device, double=False, kbs=None, verbose=True):
    with torch.no_grad():
        bpd = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            bpd += mc_bpd_batched(model, x, k, kbs)[0] * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i + 1, len(data_loader)), bpd / count, end='\r')
    return bpd / count
