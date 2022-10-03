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

import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from HAPiCLR.args.setup import parse_args_umap
from HAPiCLR.methods import METHODS
from HAPiCLR.utils.auto_umap import OfflineUMAP
from HAPiCLR.utils.classification_dataloader import prepare_data
from HAPiCLR.utils.backbones import (
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
from torchvision.models import resnet18, resnet50


def main():
    args = parse_args_umap()
    my_backbone="resnet50"
    # build paths
    backbone_model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
    }[my_backbone]

    # initialize backbone
    kwargs = {
        "cifar": False,  # <-- change this if you are running on cifar
        # "img_size": 224,  # <-- uncomment this when using vit/swin
        # "patch_size": 16,  # <-- uncomment this when using vit
    }
    cifar = kwargs.pop("cifar", False)
    # swin specific
    if "swin" in my_backbone and cifar:
        kwargs["window_size"] = 4

    model = backbone_model(**kwargs)
    if "resnet" in my_backbone:
        # remove fc layer
        model.fc = nn.Identity()
        if cifar:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            model.maxpool = nn.Identity()

    ckpt_path = args.pretrained_checkpoint_dir

    state = torch.load(ckpt_path,map_location={'cuda:7':'cuda:0'})["state_dict"]
    for k in list(state.keys()):
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    model.load_state_dict(state, strict=False)

    print(f"loaded {ckpt_path}")

    # prepare data
    train_loader, val_loader = prepare_data(
        args.dataset,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_class_num = args.subset_class_num,
        subset_rate = args.subset_rate,
    )

    umap = OfflineUMAP()

    # move model to the gpu
    device = "cuda:0"
    model = model.to(device)

    umap.plot(device, model, train_loader, "im100_train_umap.pdf")
    umap.plot(device, model, val_loader, "im100_val_umap.pdf")


if __name__ == "__main__":
    main()
