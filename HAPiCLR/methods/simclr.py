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
from HAPiCLR.losses.simclr import simclr_loss_func
from HAPiCLR.methods.base import BaseMethod
import math

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

class SimCLR(BaseMethod):
    def __init__(self, proj_output_dim: int, proj_hidden_dim: int, temperature: float, **kwargs):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(**kwargs)

        self.temperature = temperature

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimCLR, SimCLR).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
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

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        return {**out, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        feats = out["feats"]#[z1,z2]

        z = [self.projector(f) for f in feats]

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        # nce_loss = simclr_loss_func(
        #     z,
        #     indexes=indexes,
        #     temperature=self.temperature,
        # )
        nce_loss = self.nt_xent_loss(z[0],z[1],self.temperature)
        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss

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
        print(out.shape,out_dist.shape)
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss
