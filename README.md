# Multi-Task-GAN

## Introduction

This repository contains an implementation of MTL-GAN. Here we introduce several auxilary tasks to improve the quality of the generated images. 

## Requirements
```
torch==1.13.0 or newer
torchvision==0.11.3 or newer
numpy==1.23.3 or newer
pandas==1.5.2 or newer
```

All dependencies can be installed using:

```
pip install -r requirements.txt
```

## How to use

### Training

To train run the below; 

```
python main.py 
```


### Hardware

## Full documentation

### Data
```
--dataset               Dataset name ['Cifar100']

```
### Optimizer
```
--optimizer             Optimizer for model['Adam']

--lr_heads              learning rate for the heads module['0.0002']

--lr_generator          learning rate for the generator['0.0002']

--sgd_momentum          momentum parameter for SGD [0.]

--beta_1                first parameter of Adam optimizer [.5]

--beta_2                second parameter of Adam optimizer [.9]

--weight_decay          weight decay [0.]

```
## Resources

Unless stated otherwise the code in this repo is original