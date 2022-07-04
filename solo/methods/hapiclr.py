# Copyright 2022 TranNhiem & HonHai SSL development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import argparse
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import numpy as np
import math
from solo.losses.mscrl import simclr_loss_func, mscrl_loss_func_V1, mscrl_loss_func_V2, mscrl_loss_func_V3, mscrl_loss_func_V4, pixel_lavel_ontrastive, nt_xent_loss, DCL_loss_func, pixel_lavel_ontrastive_DCL, pixel_lavel_ontrastive_new
from solo.losses.byol import byol_loss_func
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
import torch.nn.functional as F
import torch.distributed as dist

#************************************************************
# SyncFunction adding to gather all the batch tensors from others GPUs
#************************************************************

class SyncFunction(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.GA = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

    def forward(self, X):
        mask = None
        if type(X) == list:
            mask = X[1]
            X = X[0]

        if mask is None:
            X = self.GA(X)
        else:
            # print(X.shape)
            # print(mask.shape)
            X = X.view(X.shape[0], X.shape[1], -1)
            mask = mask.view(mask.shape[0], mask.shape[1], -1)
            nelements = mask.sum(dim=-1)+1
            X = X.sum(dim=-1) / nelements

        X = torch.flatten(X, 1)
        return X

## Masking Steps Between Mask and Image
class Indexer(nn.Module):
    def __init__(self):
        super(Indexer, self).__init__()
    def forward(self, X, M_f, M_b):
        """Indicating the foreground and background feature.
        Args:
            X (torch.Tensor): batch of images in tensor format.
            M_f (torch.Tensor) : batch of foreground mask
            M_b (torch.Tensor) : batch of background mask
        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        # feature_f = torch.mul(X, M_f)
        # # out['foreground_feature'] = self.downsample_f(out['foreground_feature'])
        # feature_b = torch.mul(X, M_b)
        # # out['background_feature'] = self.downsample_b(out['background_feature'])

        feature_f = torch.mul(X , M_f)
        feature_b = torch.mul(X, M_b)

        return feature_f, feature_b
###  Two Sucessive Conv 1x1 Layers (reduce the dimension of the channels)
class ConvMLP(nn.Module):
    def __init__(self, chan=2048, chan_out = 256, inner_dim = 2048, scale_factor=None):
        super().__init__()
        if scale_factor != None:
            self.net = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
                nn.Conv2d(chan, inner_dim, 1),
                nn.BatchNorm2d(inner_dim),
                nn.ReLU(),
                nn.Conv2d(inner_dim, chan_out, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(chan, inner_dim, 1),
                nn.BatchNorm2d(inner_dim),
                nn.ReLU(),
                nn.Conv2d(inner_dim, chan_out, 1)
            )

    def forward(self, x):
        x = self.net(x)
        #x = torch.flatten(x, 2)
        return x

loss_types = ['V0','V1','V2','V3','V4','pixel_lavel_ontrastive']
