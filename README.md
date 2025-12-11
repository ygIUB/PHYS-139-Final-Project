# PHYS-139 Final Project: Mitochondrial Segmentation on EM Cell Images

## Introduction
This project is an effort at extending our chosen paper on MoDL to segment mitochondria from background in Electron Microscopy Images instead of Fluorescent images. Mitochondrial segmentation allows one to further study the morphology and efficiency of the mitochondria as well. This task has been carried out by using Image Preprocessing, a 2D Convolutional Neural Network and a modified U-Net model. It has been trained and validated on a dataset from CEM-MitoLab.

```text
PHYS-139-Final-Project/
├── .ipynb_checkpoints/
├── MitoSegNet/     <-- MitoSegNet model files
├── MoDL_pre/       <-- MoDL model from original paper 
├── MoDL_seg/       <-- MoDL files modified for benchmark testing
├── benchmarking/   <-- Benchmarking scripts
├── helpers/        <-- Helper functions for our model
├── .gitignore
├── LICENSE
├── README.md
├── ourResUnet.py   <-- Our Model's main file
├── requirements.docker.txt  <-- Requirements.txt for Docker
├── requirements.txt         <-- Requirements.txt
└── requirements.win.txt     <-- Requirements.txt for Windows
```

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)

## Requirements
The minimal requirements for this setup is unknown but the machine running our model had the specifications as follows.

OS: Windows 11 <br>
Python: 3.10 (Conda env: modl-gpu-win) <br>
GPU: NVIDIA RTX 5080 (16 GB VRAM) + integrated AMD GPU <br>
RAM: 32 GB <br>
Framework: TensorFlow 2.10 + tensorflow-directml-plugin (so TF runs on GPU via DirectML)

## Installation
To install and run our model, the GitHub Repository can be cloned and the essential requirements can be installed according to the environment using <br>
1. ```git clone git@github.com:ygIUB/PHYS-139-Final-Project```
2. ```cd PHYS-139-Final-Project```
3. ```conda create -n modl-gpu-win```
4. ```conda activate modl-gpu-win```
5. ```pip install --upgrade pip```
3. ```pip install -r requirements.txt```

The dataset must be placed in the directory PHYS-139-Final-Project/. Please notice that you require Anaconda or Miniconda as a prerequisite.

## Usage
In your Environment, you first need to run <br>
```python pack_512_only.py``` <br>

This picks and pairs all the 512x512 images from the dataset. 

Please make sure at the bottom of Modl_seg/train.py you have
```
if __name__ == '__main__':
    myunet = myUnet(model_type="baseline") or 'our'; 'baseline' means ResUnet and 'our' refers to OurResUnet
    myunet.train
```

Then, you can directly train with train.py as<br>
```python train.py```<br>

The result of the model would be stored in folder **model/**

## Data
The Project's dataset has been referenced in the GitHub repository. It is saved in the form of a .zip file in a Google Drive. The link is [here](https://drive.google.com/file/d/1ZmZ1RG796ClDXdjM_TKP6RGAd-pNKZfH/view?usp=drive_link).

A good way to procure the dataset would be to directly download it in the same directory as the cloned directory and then unzip it. <br>
```unzip cem-mitolab.zip```