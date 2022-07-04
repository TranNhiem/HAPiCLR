# Copyright 2021 solo-learn development team.

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

# class SyncFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, tensor):
#         ctx.batch_size = tensor.shape[0]
#         gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
#         torch.distributed.all_gather(gathered_tensor, tensor)
#         gathered_tensor = torch.cat(gathered_tensor, 0)
#         return gathered_tensor
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
#         idx_from = torch.distributed.get_rank() * ctx.batch_size
#         idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
#         return grad_input[idx_from:idx_to]

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

class MSCRL(BaseMethod):
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

        if "byol" in self.loss_type:
            proj_output_dim = proj_hidden_dim
        # projector
        self.projector = nn.Sequential(
            Downsample(),
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.downsample = nn.Sequential(
            Downsample()
        )

        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.indexer = Indexer()
        self.convMLP = ConvMLP(self.features_dim, pixel_hidden_dim, pixel_output_dim, None)


    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(MSCRL, MSCRL).add_model_specific_args(parent_parser)
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
        z = self.projector(out["feats"])
        z_f , z_b = self.indexer(out["feats"])
        z_f = self.projector(z_f)
        z_b = self.projector(z_b)

        return {**out, "z": z, "z_f": z_f, "z_b": z_b}



    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes, batch_, M_f, M_b = batch

        out = super().training_step(batch_, batch_idx)
        class_loss = out["loss"]

        feats = out["feats"]
        middle_feats = out["middle_feats"]
        if "byol" in self.loss_type or "barlow" in self.loss_type or "xent" in self.loss_type:
            z = [self.projector(f) for f in feats]
        else:
            z = torch.cat([self.projector(f) for f in feats])
        # z = [self.projector(f) for f in feats]

        if "pixel_level_contrastive" in self.loss_type:
            #print("pixel_feature")
            if "cat" in self.loss_type:
                if self.scale_factor is not None:
                    tmp_feats = [self.upsample(feat) for feat in feats]
                else:
                    tmp_feats = feats
                middle_feats = [torch.cat([middle_feat,tmp_feat],1)for middle_feat,tmp_feat in zip(middle_feats,tmp_feats)]
            pixel_feature = [self.convMLP(f) for f in middle_feats]
        else:
            pixel_feature = feats

        if "feature_contrastive" in self.loss_type:
            z_f = []
            z_b = []
            for f, m_f, m_b in zip(middle_feats,M_f,M_b):
                a,b = self.indexer(f,m_f,m_b)
                z_f.append(a)
                z_b.append(b)

            z_f = torch.cat([self.downsample([f, m]) for f, m in zip(z_f, M_f)])
            z_b = torch.cat([self.downsample([f, m]) for f, m in zip(z_b, M_b)])

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        simclr_loss = 0.0
        pixel_levels_loss = 0.0
        pos = 0.0
        neg = 0.0
        pixel_pos = 0.0
        pixel_neg = 0.0
        if "xent" in self.loss_type:
            temp, pos, neg = nt_xent_loss(
                z[0],
                z[1],
                temperature=self.temperature,
            )
            simclr_loss += temp
        if "byol" in self.loss_type :
            neg_cos_sim = 0
            for v1 in range(self.num_large_crops):
                for v2 in np.delete(range(self.num_crops), v1):
                    neg_cos_sim += byol_loss_func(z[v2], self.downsample(feats[v1]))
            simclr_loss += neg_cos_sim
        if "barlow" in self.loss_type :
            temp = barlow_loss_func(z[0],z[1])
            simclr_loss += temp
        if "foreground_loss" in self.loss_type :
            temp, pos, neg = simclr_loss_func(
                z_f,
                indexes=indexes,
                temperature=self.temperature,
            )
            simclr_loss += temp
        if "background_loss" in self.loss_type :
            temp, pos, neg = simclr_loss_func(
                z_b,
                indexes=indexes,
                temperature=self.temperature,
            )
            simclr_loss += temp
        if "V0" in self.loss_type :
            temp, pos, neg = simclr_loss_func(
                z,
                indexes=indexes,
                temperature=self.temperature,
            )
            # temp, pos, neg = self.nt_xent_loss(
            #     z[0],
            #     z[1],
            #     temperature=self.temperature,
            # )
            simclr_loss += temp
        if "V1" in self.loss_type:
            temp, pos, neg = mscrl_loss_func_V1(
                z, z_f ,z_b,
                indexes=indexes,
                temperature=self.temperature,
            )
            simclr_loss += temp
        if "V2" in self.loss_type:
            temp, pos, neg = mscrl_loss_func_V2(
                z, z_f ,z_b,
                indexes=indexes,
                temperature=self.temperature,
            )
            simclr_loss += temp
        if "V3" in self.loss_type :
            temp, pos, neg= mscrl_loss_func_V3(
                z, z_f ,z_b,
                indexes=indexes,
                temperature=self.temperature,
                alpha=self.alpha,
            )
            simclr_loss += temp
        if "V4" in self.loss_type:
            temp, pos, neg = mscrl_loss_func_V4(
                z, z_f ,z_b,
                indexes=indexes,
                temperature=self.temperature,
            )
            simclr_loss += temp
        if "DCL" in self.loss_type:
            temp, pos, neg = DCL_loss_func(
                z,
                indexes=indexes,
                temperature=self.temperature,
            )
            simclr_loss += temp

        if "pixel_lavel_ontrastive_new" in self.loss_type:
            if "pixel_lavel_ontrastive_new_background" in self.loss_type:
                pixel_levels_loss, pixel_pos, pixel_neg = pixel_lavel_ontrastive_new(
                    z, [torch.flatten(m, 2) for m in pixel_feature], [torch.flatten(m, 2) for m in M_f],
                    indexes=indexes,
                    pixel_mask_b=[torch.flatten(m, 2) for m in M_b],
                    temperature=0.2,
                )
            else:
                pixel_levels_loss, pixel_pos, pixel_neg = pixel_lavel_ontrastive_new(
                    z, [torch.flatten(m, 2) for m in pixel_feature], [torch.flatten(m, 2) for m in M_f],
                    indexes=indexes,
                    temperature=0.2,
                )

        elif "pixel_level_contrastive" in self.loss_type:
            if "pixel_level_contrastive_background" in self.loss_type:
                pixel_levels_loss, pixel_pos, pixel_neg = pixel_lavel_ontrastive(
                    z, [torch.flatten(m, 2) for m in pixel_feature], [torch.flatten(m, 2) for m in M_f],
                    indexes=indexes,
                    pixel_mask_b=[torch.flatten(m, 2) for m in M_b],
                    temperature=0.2,
                )
            else:
                pixel_levels_loss, pixel_pos, pixel_neg = pixel_lavel_ontrastive(
                    z, [torch.flatten(m, 2) for m in pixel_feature], [torch.flatten(m, 2) for m in M_f],
                    indexes=indexes,
                    temperature=0.2,
                )

        if "pixel_level_contrastive_background_DCL" in self.loss_type:
            pixel_levels_loss, pixel_pos, pixel_neg = pixel_lavel_ontrastive_DCL(
                z, [torch.flatten(m, 2) for m in pixel_feature], [torch.flatten(m, 2) for m in M_f],
                indexes=indexes,
                pixel_mask_b=[torch.flatten(m, 2) for m in M_b],
                temperature=0.2)

        if self.alpha is None:
            msc_loss = simclr_loss + pixel_levels_loss
        else:
            msc_loss = simclr_loss + self.alpha*pixel_levels_loss

        metrics = {"train_MSC_loss": msc_loss, "train_pos_loss": pos, "train_neg_loss": neg, "pixel_pos": pixel_pos, "pixel_neg": pixel_neg,
                   "simcllr_loss": simclr_loss, "pixel_levels_loss": pixel_levels_loss}
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        #self.log("train_MSC_loss", msc_loss, on_epoch=True, sync_dist=True)

        return msc_loss + class_loss
