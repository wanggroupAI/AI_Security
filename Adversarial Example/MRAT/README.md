
This repository contains the code for the paper:

[Towards Defending Adversarial Patch Attacks with Mask-Reconstruction-assisted Adversarial Training]


## Requirements

*   Python 3.7.12

*   torch 1.12.1

*   torchvision 0.13.1

*   numpy 1.21.6

## Experiments

#### Introduction

*   `attackr.py` : Implements adversarial example generation using adversarial masks.

*   `Trainer.py` : Defines the Trainer class responsible for model training, including loss computation, optimization, and evaluation loops.
   
*   `train.py` : The main entry point for training. It parses arguments, initializes datasets and models, and launches the training process.

    `models` :  Contains definitions of the MRAT model architectures, including ResNet, DenseNet, VGG, and WideResNet variants.

    `utils` : Provides utility functions for dataset loading, progress visualization, and other helper routines.

#### Example Usage

    python train.py --model resnet18 --dataset imagenette --block_size 32 --mask_ratio 0.4 --split 0.5 --batch_size 128 --device cuda:0
