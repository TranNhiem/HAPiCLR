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

from torch import Tensor, nn
import numpy as np
import math
from solo.losses.mscrl import simclr_loss_func, mscrl_loss_func_V1, mscrl_loss_func_V2, mscrl_loss_func_V3, mscrl_loss_func_V4, pixel_lavel_ontrastive, nt_xent_loss, DCL_loss_func, pixel_lavel_ontrastive_DCL, pixel_lavel_ontrastive_new
from solo.losses.byol import byol_loss_func
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from torchvision.models import resnet18, resnet50
import torch.nn.functional as F
import torch.distributed as dist

#************************************************************
# SyncFunction adding to gather all the batch tensors from others GPUs
#************************************************************



class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


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

class HAPiCLR(BaseMethod): 
    def __init__(self, proj_output_dim: int, proj_hidden_dim: int, pixel_hidden_dim: int, pixel_output_dim: int, temperature: float, loss_type: str, alpha: int = None, scale_factor=None, **kwargs):
        """Implements MSCRL.
        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            loss_type (str): which loss need to use]
        """

        super().__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 0.5
        #assert loss_type in loss_types, "Loss type didn't included"
        self.loss_type = loss_type

        #***********************
        # MLP projector
        #**********************
        self.projector = nn.Sequential(
            Downsample(),
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.downsample = nn.Sequential(
            Downsample()
        )
        #**********************
        # Conv 1x1  projector
        #**********************
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.indexer = Indexer()
        self.convMLP = ConvMLP(self.features_dim, pixel_hidden_dim, pixel_output_dim, None)


    #************************************
    # Adding default arugments for models
    #************************************
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(HAPiCLR, HAPiCLR).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mscrl")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--pixel_output_dim", type=int, default=256)
        parser.add_argument("--pixel_hidden_dim", type=int, default=2048)
        parser.add_argument("--loss_type", type=str, default="byol+f_loss+b_loss")

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--beta", type=str, default="0.5")
        parser.add_argument("--scale_factor", type=int, default=None)

        return parent_parser
    
    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()},
                                  {"params": self.convMLP.parameters()}
                                  ]
        return super().learnable_params + extra_learnable_params

        
        
    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """
        _, X_, M_f, M_b = X

        out = super().forward(X_, *args, **kwargs)
        ## Output representation [batch,X1], [batch,X2]
        z = self.projector(out["feats"])
        z_f, z_b = self.indexer(out["feats"])

        z_f = self.projector(z_f)
        z_b = self.projector(z_b)
        
        return {**out, "z": z, "z_f": z_f, "z_b": z_b}
    
    def shared_step(self, batch, batch_idx):
        
        """Training step for SimCLR reusing BaseMethod training step.
        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
            [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.
        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """
        
        indexes, X, M_f, M_b = batch
        out = super().training_step(X, batch_idx)
        class_loss = out["loss"]

        feats = out["feats"]
        #print(feats.shape)
        
        z = [self.projector(f) for f in feats]
        #z[[b, x1], [b, x2]]
        #print(z.shape)
        # get projection representations
        z1 = z[0]
        #print(z1.shape)
        z2 = z[1]
        loss = self.nt_xent_loss(z1, z2, self.temperature)
        
      
        return loss , class_loss


    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor: 
        loss, class_loss= self.shared_step(batch, batch_idx) 
        metrics={
            "Contrastive_loss": loss
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
       
        return loss + class_loss

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            print("Distributed_multi_GPUs")
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)

        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss
