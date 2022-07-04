
from typing import Type, Any, Callable, Union, List, Optional
from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.utils.backbones import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    poolformer_m36,
    poolformer_m48,
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
    resnet18_MNCRL,
    resnet50_MNCRL,
)
from solo.utils.knn import WeightedKNNClassifier
from solo.utils.lars import LARSWrapper
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torchvision.models import resnet18, resnet50
import composer.functional as cf

def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


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




class HAPiCLR_Unified(pl.LightningModule):

    _SUPPORTED_BACKBONES = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "resnet18_MNCRL": resnet18_MNCRL,
        "resnet50_MNCRL": resnet50_MNCRL,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
        "poolformer_s12": poolformer_s12,
        "poolformer_s24": poolformer_s24,
        "poolformer_s36": poolformer_s36,
        "poolformer_m36": poolformer_m36,
        "poolformer_m48": poolformer_m48,
        "convnext_tiny": convnext_tiny,
        "convnext_small": convnext_small,
        "convnext_base": convnext_base,
        "convnext_large": convnext_large,
    }

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        backbone_args: dict,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        weight_decay: float,
        classifier_lr: float,
        exclude_bias_n_norm: bool,
        accumulate_grad_batches: Union[int, None],
        extra_optimizer_args: Dict,
        scheduler: str,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        num_large_crops: int,
        num_small_crops: int,
        eta_lars: float = 1e-3,
        grad_clip_lars: bool = False,
        lr_decay_steps: Sequence = None,
        knn_eval: bool = False,
        knn_k: int = 20,
        channels_last: bool = False,
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            backbone (str): architecture of the base backbone.
            num_classes (int): number of classes.
            backbone_params (dict): dict containing extra backbone args, namely:
                #! optional, if it's not present, it is considered as False
                cifar (bool): flag indicating if cifar is being used.
                #! only for resnet
                zero_init_residual (bool): change the initialization of the resnet backbone.
                #! only for vit
                patch_size (int): size of the patches for ViT.
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lars (bool): flag indicating if lars should be used.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            classifier_lr (float): learning rate for the online linear classifier.
            exclude_bias_n_norm (bool): flag indicating if bias and norms should be excluded from
                lars.
            accumulate_grad_batches (Union[int, None]): number of batches for gradient accumulation.
            extra_optimizer_args (Dict): extra named arguments for the optimizer.
            scheduler (str): name of the scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            num_large_crops (int): number of big crops.
            num_small_crops (int): number of small crops .
            eta_lars (float): eta parameter for lars.
            grad_clip_lars (bool): whether to clip the gradients in lars.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
            knn_eval (bool): enables online knn evaluation while training.
            knn_k (int): the number of neighbors to use for knn.

        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly based on the
            batch size and gradient accumulation.

        .. note::
            For CIFAR10/100, the first convolutional and maxpooling layers of the ResNet backbone
            are slightly adjusted to handle lower resolution images (32x32 instead of 224x224).

        """

        super().__init__()

        # resnet backbone related
        self.backbone_args = backbone_args
        # print(self.backbone_args)

        # training related
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.classifier_lr = classifier_lr
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.num_large_crops = num_large_crops
        self.num_small_crops = num_small_crops
        self.eta_lars = eta_lars
        self.grad_clip_lars = grad_clip_lars
        self.knn_eval = knn_eval
        self.knn_k = knn_k
        self.channels_last = channels_last

        # multicrop
        self.num_crops = self.num_large_crops + self.num_small_crops
        # all the other parameters
        self.extra_args = kwargs
        print("num_class:",self.num_classes)
        # turn on multicrop if there are small crops
        self.multicrop = self.num_small_crops != 0
        
        ##*************************************
        # if accumulating gradient then scale lr
        ##*************************************
        if self.accumulate_grad_batches:
            self.lr = self.lr * self.accumulate_grad_batches
            self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
            self.min_lr = self.min_lr * self.accumulate_grad_batches
            self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

       
        ##*************************************
        ## Get the Backbone Architecture
        ##*************************************
       
        assert backbone in BaseMethod._SUPPORTED_BACKBONES
        self.base_model = self._SUPPORTED_BACKBONES[backbone]
        self.backbone_name = backbone

        # initialize backbone
        kwargs = self.backbone_args.copy()
        cifar = kwargs.pop("cifar", False)
        # swin specific
        if "swin" in self.backbone_name and cifar:
            kwargs["window_size"] = 4

        self.backbone = self.base_model(**kwargs)
        
        if "resnet" in self.backbone_name:
            
            self.features_dim = self.backbone.inplanes
            self.backbone.fc = nn.Identity()
            
            if cifar:
                self.backbone.conv1 = nn.Conv2d( 3, 64, kernel_size=3, stride=1, padding=2, bias=False)
                self.backbone.maxpool = nn.Identity()
        else:
            self.features_dim = self.backbone.num_features

        if "MNCRL" in self.backbone_name:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(self.features_dim, self.num_classes),
            )
        else:
            self.classifier = nn.Linear(self.features_dim, self.num_classes)
        ##*************************************
        # MLP projection Head 
        ##*************************************

        self.projector = nn.Sequential(
            Downsample(),
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        self.downsample = nn.Sequential( Downsample())
        ##*************************************
        # Conv 1*1 projection Head 
        ##*************************************

        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.indexer = Indexer()
        self.convMLP = ConvMLP(self.features_dim, pixel_hidden_dim, pixel_output_dim, None)

        if self.knn_eval:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx="euclidean")
        if self.channels_last:
            cf.apply_channels_last(self.backbone)
            cf.apply_channels_last(self.classifier)
    
    
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("base")

        # backbone args
        SUPPORTED_BACKBONES = BaseMethod._SUPPORTED_BACKBONES

        parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, type=str)
        # extra args for resnet
        parser.add_argument("--zero_init_residual", action="store_true")
        # extra args for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam", "adamw"]

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=1e-3, type=float)
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "reduce",
            "cosine",
            "warmup_cosine",
            "step",
            "exponential",
            "none",
        ]

        parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.00003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        # MLP Projector and Conv 1*1 Projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--pixel_output_dim", type=int, default=256)
        parser.add_argument("--pixel_hidden_dim", type=int, default=2048)
        parser.add_argument("--loss_type", type=str, default="byol+f_loss+b_loss")

        # For Contrastive loss
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--beta", type=str, default="0.5")
        parser.add_argument("--scale_factor", type=int, default=None)

        # DALI only
        # uses sample indexes as labels and then gets the labels from a lookup table
        # this may use more CPU memory, so just use when needed.
        parser.add_argument("--encode_indexes_into_labels", action="store_true")

        # online knn eval
        parser.add_argument("--knn_eval", action="store_true")
        parser.add_argument("--knn_k", default=20, type=int)

        return parent_parser

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
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

    def forward(self, *args, **kwargs) -> Dict:
        """Dummy forward, calls base forward."""
        return self.base_forward(*args, **kwargs)

    def base_forward(self, X: torch.Tensor) -> Dict:
        """Basic forward that allows children classes to override forward().

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """
        try:
            middle_feats, feats = self.backbone(X)
        except ValueError:
            feats = self.backbone(X)
            middle_feats = feats
        
        logits = self.classifier(feats.detach())
        return {
            "logits": logits,
            "feats": feats,
            "middle_feats": middle_feats, }


    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self.base_forward(X)
        logits = out["logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # handle when the number of classes is smaller than 5
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        return {**out, "loss": loss, "acc1": acc1, "acc5": acc5}

    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """

        try:
            _, X, M_f, M_b, targets = batch
        
        except ValueError:
            X, targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops

        outs = [self._base_shared_step(x, targets) for x in X[: self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        if self.multicrop:
            outs["feats"].extend([self.backbone(x) for x in X[self.num_large_crops :]])
        # loss and stats
        outs["loss"] = sum(outs["loss"]) / self.num_large_crops
        outs["acc1"] = sum(outs["acc1"]) / self.num_large_crops
        outs["acc5"] = sum(outs["acc5"]) / self.num_large_crops

        feats=out["feats"]
        z = torch.cat([self.projector(f) for f in feats])
        z1= z[0]
        z2= z[1]
        contrastive_loss = self.nt_xent_loss(z1, z2, self.temperature)
        metrics = {
            "train_class_loss": outs["loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],   
            "contrastive_loss": contrastive_loss, 
            }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if self.knn_eval:
            targets = targets.repeat(self.num_large_crops)
            mask = targets != -1
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_large_crops])[mask].detach(),
                train_targets=targets[mask],
            )

        return contrastive_loss

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        X, targets = batch
        batch_size = targets.size(0)

        out = self._base_shared_step(X, targets)

        if self.knn_eval and not self.trainer.sanity_checking:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        if self.knn_eval and not self.trainer.sanity_checking:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)

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