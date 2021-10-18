import torch


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def BarlowTwinsLoss(z1, z2, batch_size, scale_loss=1.0 / 32, lambd=3.9e-3):
    """
    Zbontar et al., Barlow Twins: Self-Supervised Learning via Redundancy Reduction
    https://arxiv.org/abs/2103.03230

    Implementation from https://github.com/facebookresearch/barlowtwins
    Copyright (c) Facebook, Inc. and its affiliates.
    """
    # empirical cross-correlation matrix
    c = z1.T @ z2

    # sum the cross-correlation matrix between all gpus
    c.div_(batch_size)
    # torch.distributed.all_reduce(c)

    # use --scale-loss to multiply the loss by a constant factor
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(scale_loss)
    off_diag = off_diagonal(c).pow_(2).sum().mul(scale_loss)
    loss = on_diag + lambd * off_diag

    return loss
