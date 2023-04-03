# HAPiCLR - Heuristic Attention Pixel-level Contrastive Loss Representation 



<span style="color: red"><strong> </strong></span> This is offical implemenation of HAPiCLR framework</a>.


<div align="center">
  <img width="100%" alt="HARL Framework Illustration" src="images/HAPiCLR_framework.gif">
</div>
<div align="center">
  End-to-End HAPiCLR Framework (from <a href="">our blog here</a>).
</div>

# Table of Contents

  - [Installation](#installation)
  - [Dataset -- MASK Generator](#Generating-Heuristic-binary-mask-Natural-image)
  - [Configure Self-Supervised Pretraining](#Setup-self-supervised-pretraining)
    - [Dataset](#Natural-Image-Dataset)
    - [Hyperamters Setting](#Important-Hyperparameter-Setting)
    - [Choosing # augmentation Strategies](#Number-Augmentation-Strategies)
    - [Single or Multi GPUs](#Single-Multi-GPUS)
  - [Pretrained model](#model-weights)
  - [Downstream Tasks](#running-tests)
     - [Image Classification Tasks](#Natural-Image-Classification)
     - [Other Vision Tasks](#Object-Detection-Segmentation)
  - [Contributing](#contributing)

## Installation

```
pip or conda installs these dependents in your local machine
```
### Requirements
* torch
* torchvision
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts
* torchmetrics
* scipy
* timm

## Dataset -- Heuristic Mask Retrieval Techniques

```
If you are using ImageNet 1k dataset for self-supervised pretraining. 
We provodie two sets of heuristic mask generated for whole ImageNet train set available for download. 

```
|                         Dataset Mask Generator                               |
|---------------------------------------------------------------------------------|
|[Heuristic Mask for ImageNet 1K Train set](https://drive.google.com/file/d/1-Ph6f4lLVe9Og_6_Ko2vx4sSDYKO6b7C/view?usp=sharing)|

### Using Custome Dataset 
**1.Generating Heuristic Binary Mask Using Deep Learning method**

We created one python module that directly with the input directory of your dataset
then generate by providing the filename

'''
/heuristic_mask_techniques/Deeplearning_methods/DeepMask.py

'''

## Self-supervised Pretraining

###  Preparing  Dataset: 

**NOTE:** Currently, This repository support self-supervised pretraining on the ImageNet dataset. 
+ 1. Download ImageNet-1K dataset (https://www.image-net.org/download.php). Then unzip folder follow imageNet folder structure. 


###  in pretraining Flags: 
`
Naviaging to the 

bash_files/pretrain/imagenet/HARL.sh
`

**1 Changing the dataset directory according to your path**
    `
    --train_dir ILSVRC2012/train \
    --val_dir ILSVRC2012/val \
    --mask_dir train_binary_mask_by_USS \
    `
**2 Other Hyperparameters setting** 
  
  - Use a large init learning rate {0.3, 0.4} for `short training epochs`. This would archieve better performance, which could be hidden by the initialization if the learning rate is too small.Use a small init learning rate for Longer training epochs should use value around 0.2.

    `
    --max_epochs 100 \
    --batch_size 512 \
    --lr 0.5 \
    `
**3 Distributed training in 1 Note**

`
Controlling number of GPUs in your machine by change the --gpus flag
    --gpus 0,1,2,3,4,5,6,7\
    --accelerator gpu \
    --strategy ddp \
`
## HARL Pre-trained models  

We opensourced total 8 pretrained models here, corresponding to those in Table 1 of the <a href="">HARL</a> paper:

|   Depth | Width   | SK    |   Param (M)  | Pretrained epochs| SSL pretrained learning_rate |Projection head MLP Dimension| Heuristic Mask| Linear eval  |
|--------:|--------:|------:|--------:|-------------:|--------------:|--------:|-------------:|-------------:|
| [ResNet50 (1x)]() | 1X | False | 24 | 100 |  0.5| 256 |Deep Learning mask| ## |     
| [ResNet50 (1x)]() | 1X  | False | 24 | 100 |  0.3 | 256 |Deep Learning mask|  ## |   


These checkpoints are stored in Google Drive Storage:

## Finetuning the linear head (linear eval)

To fine-tune a linear head (with a single GPU), try the following command:

For fine-tuning a linear head on ImageNet using GPUs, first set the `CHKPT_DIR` to pretrained model dir and set a new `MODEL_DIR`, then use the following command:
`
Stay tune! The instructions will update soon
`

## Semi-supervised learning and fine-tuning the whole network

You can access 1% and 10% ImageNet subsets used for semi-supervised learning via [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_subset): simply set `dataset=imagenet2012_subset/1pct` and `dataset=imagenet2012_subset/10pct` in the command line for fine-tuning on these subsets.

You can also find image IDs of these subsets in `imagenet_subsets/`.

To fine-tune the whole network on ImageNet (1% of labels), refer to the following command:

`
Stay tune! The instructions will update soon
`

## Other resources
update soon

## Known issues

* **Pretrained models / Checkpoints**: Multi-GPUs training decreasing performance --> 

## Citation

`
@article{T,
  title={Heuristic Attention Pixel-level Contrastive Loss Representation Learning for Self-Supervised Pretraining},
  author={},
  journal={},
  year={2022},
  volume={}
}
`

