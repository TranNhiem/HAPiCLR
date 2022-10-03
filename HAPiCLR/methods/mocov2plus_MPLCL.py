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
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from HAPiCLR.losses.moco import moco_loss_func
from HAPiCLR.losses.mscrl import pixel_lavel_ontrastive, pixel_lavel_ontrastive_new
from HAPiCLR.methods.base import BaseMomentumMethod
from HAPiCLR.utils.momentum import initialize_momentum_params
from HAPiCLR.utils.misc import gather

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
    def __init__(self, chan=2048, chan_out = 256, inner_dim = 2048):
        super().__init__()
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

class MoCoV2Plus_MPLCL(BaseMomentumMethod):
    queue: torch.Tensor

    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        queue_size: int,
        pixel_hidden_dim: int,
        pixel_output_dim: int,
        **kwargs
    ):
        """Implements MoCo V2+ (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            queue_size (int): number of samples to keep in the queue.
        """

        super().__init__(**kwargs)

        self.temperature = temperature
        self.queue_size = queue_size

        # projector
        self.projector = nn.Sequential(
            Downsample(),
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),

        )
        self.convMLP = ConvMLP(self.features_dim, pixel_hidden_dim, pixel_output_dim)

        # momentum projector
        self.momentum_projector = nn.Sequential(
            Downsample(),
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        self.momentum_convMLP = ConvMLP(self.features_dim, pixel_hidden_dim, pixel_output_dim)

        initialize_momentum_params(self.projector, self.momentum_projector)
        initialize_momentum_params(self.convMLP, self.momentum_convMLP)

# create the queue
        self.register_buffer("queue", torch.randn(2, proj_output_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(MoCoV2Plus_MPLCL, MoCoV2Plus_MPLCL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mocov2plus")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--pred_hidden_dim", type=int, default=2048)
        parser.add_argument("--pixel_output_dim", type=int, default=256)
        parser.add_argument("--pixel_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--scale_factor", type=int, default=None)
        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)


        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters together with parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()},
                                  {"params": self.convMLP.parameters()}
                                  ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector),
                                (self.convMLP, self.momentum_convMLP)]
        return super().momentum_pairs + extra_momentum_pairs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner.

        Args:
            keys (torch.Tensor): output features of the momentum backbone.
        """

        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the online backbone and projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = F.normalize(self.projector(out["feats"]), dim=-1)
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for MoCo reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the
                format of [img_indexes, [X], Y], where [X] is a list of size self.num_large_crops
                containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MOCO loss and classification loss.

        """
        indexes, batch_, M_f, M_b = batch
        out = super().training_step(batch_, batch_idx)
        class_loss = out["loss"]
        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]


        q1 = self.projector(feats1)
        q2 = self.projector(feats2)
        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)

        pixel_feature = [self.convMLP(f) for f in out["feats"]]

        with torch.no_grad():
            k1 = self.momentum_projector(momentum_feats1)
            k2 = self.momentum_projector(momentum_feats2)
            k1 = F.normalize(k1, dim=-1)
            k2 = F.normalize(k2, dim=-1)

            momentum_f1 = self.momentum_convMLP(momentum_feats1)
            momentum_f2 = self.momentum_convMLP(momentum_feats2)

        # ------- contrastive loss -------
        # symmetric
        queue = self.queue.clone().detach()
        nce_loss = (
            moco_loss_func(q1, k2, queue[1], self.temperature)
            + moco_loss_func(q2, k1, queue[0], self.temperature)
        ) / 2

        pixel_levels_loss1, pixel_pos1, pixel_neg1 = pixel_lavel_ontrastive_new(None, [torch.flatten(pixel_feature[0], 2), torch.flatten(momentum_f2, 2)],
                                                                         [torch.flatten(m, 2) for m in M_f], None, [torch.flatten(m, 2) for m in M_b], 0.2)
        pixel_levels_loss2, pixel_pos2, pixel_neg2 = pixel_lavel_ontrastive_new(None, [ torch.flatten(momentum_f1, 2), torch.flatten(pixel_feature[1], 2)],
                                                                         [torch.flatten(m, 2) for m in M_f], None, [torch.flatten(m, 2) for m in M_b], 0.2)
        pixel_levels_loss = (pixel_levels_loss1 + pixel_levels_loss2) / 2.0
        pixel_pos = (pixel_pos1 + pixel_pos2) / 2.0
        pixel_neg = (pixel_neg1 + pixel_neg2) / 2.0
        # pixel_levels_loss, pixel_pos, pixel_neg = pixel_lavel_ontrastive(None, [torch.flatten(m, 2) for m in pixel_feature],
        #                                                                  [torch.flatten(m, 2) for m in M_f], None, [torch.flatten(m, 2) for m in M_b], 0.2)
        # ------- update queue -------
        keys = torch.stack((gather(k1), gather(k2)))
        self._dequeue_and_enqueue(keys)

        self.log_dict({"train_nce_loss": nce_loss, "pixel_pos": pixel_pos, "pixel_neg": pixel_neg,
                  "pixel_levels_loss": pixel_levels_loss}, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss
