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
import torch
import numpy as np
import math
from torch import Tensor, nn
from HAPiCLR.methods.base import BaseMethod
import torch.nn.functional as F
from typing import Any, Dict, List, Sequence, Union, Tuple

from HAPiCLR.losses.nt_xent_loss import NTXentLoss
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from HAPiCLR.utils.lars import LARSWrapper
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
# from solo.utils.distributed_util import gather_from_all
# from classy_vision.generic.distributed_util import get_cuda_device_index, get_rank
# from classy_vision.losses import ClassyLoss, register_loss



from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
#************************************************************
# SyncFunction adding to gather all the batch tensors from others GPUs
#************************************************************


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
    def __init__(self, 
        # optimizer: str,
        # lars: bool,
        # lr: float,
        # weight_decay: float,
        # classifier_lr: float,
        # exclude_bias_n_norm: bool,
        # accumulate_grad_batches: Union[int, None],
        # extra_optimizer_args: Dict,
        # scheduler: str,
        # min_lr: float,
        # warmup_start_lr: float,
        # warmup_epochs: float,
    proj_output_dim: int, proj_hidden_dim: int,
     pixel_hidden_dim: int, pixel_output_dim: int, temperature: float, 
     loss_type: str, alpha: int = None,  gather_distributed_gpus: bool= True,  scale_factor=None, 

    **kwargs):
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
        self.criterion = NTXentLoss(gather_distributed=True, temperature=self.temperature)
        #***********************
        # MLP projector
        #**********************
        self.projector = nn.Sequential(
            #Downsample(),
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


        # self.optimizer = optimizer
        # self.lars = lars
        # self.lr = lr
        # self.weight_decay = weight_decay
        # self.classifier_lr = classifier_lr
        # self.exclude_bias_n_norm = exclude_bias_n_norm
        # self.accumulate_grad_batches = accumulate_grad_batches
        # self.extra_optimizer_args = extra_optimizer_args
        # self.scheduler = scheduler
        # self.lr_decay_steps = lr_decay_steps
        # self.min_lr = min_lr
        # self.warmup_start_lr = warmup_start_lr
        # self.warmup_epochs = warmup_epochs


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
        parser.add_argument("--temperature", type=float, default=0.5)
        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--beta", type=str, default="0.5")
        parser.add_argument("--scale_factor", type=int, default=None)

        # optimizer --> Inherence from base.py
        
        # scheduler --> Inherence from base.py

        # For gather embedding from other GPUs
        parser.add_argument("--gather_distributed_gpus", type=bool, default=True, help="True If training with >2 else False")

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
    
    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]


    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        # optionally wrap with lars
        if self.lars:
            assert self.optimizer == "sgd", "LARS is only compatible with SGD."
            optimizer = LARSWrapper(
                optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )

        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_epochs,
                warmup_start_lr=self.warmup_start_lr,
                eta_min=self.min_lr,
            )
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

        
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
        loss = self.criterion(z1, z2)
        #loss = self.nt_xent_loss(z1, z2, self.temperature)
        return loss , class_loss


    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor: 
        loss, class_loss= self.shared_step(batch, batch_idx) 
        metrics={
            "Contrastive_loss": loss
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
       
        return loss + class_loss

    
