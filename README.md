# MedAI
Project for UvA course Medical Imaging for AI.

## Install
Custom environment can be setup with anaconda:

``` conda env create -f environment.yml ```

Data can be downloaded by running ```get_data.sh``` in ```/scripts/``` for training and validation set respectively. 

Due to Snellius restrictions some package had to be installed using python scripts. Running this while in the 
environments installs the rest of the packages.

## Preprocessing
All 2D preprocessing is contained in ```preprocessing_2D_vanilla.py```. Running this file creates a folder called ```/dataset_model/``` which contains a folder containing ```images/``` and ```labels/```.

All 3D preprocessing ...

## Training
2D training for UNet is done by running ```train_2D.py```. 2D training for vitseg can be done by running ```vitseg_grayscale3d.py```

## Evaluation
Evaluation for 2D models is all collected in ```evaluate_2d.py```. Reports f1, dice and IOU scores for UNet and vitseg.

