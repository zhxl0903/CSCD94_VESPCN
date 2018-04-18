# CSCD94 Video Super Resolution Project

This project contains a collection of video super resolution methods including a 9L-E3-MC VESPCN Network
and a 9L Single Frame ESPCN Network along with Bicubic and SRCNN.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites:

Python 3.6 with Tensorflow 1.6 or 1.7

#### Required Python Packages:

- glob2, h5py, opencv-python, scipy, and numpy

### Setup

The following modes are currently supported by this project:

```
Mode                Description                       Training Required
 0     Spatial Trandformer Network                           Yes      
 1     Single Frame 9-Layer ESPCN                            Yes 
 2     9-Layer-Early-Fusion Motion-Compensated VESPCN        Yes
 3     Bicubic                                               No
 4     SRCNN                                                 Yes                     
 5     Multi-Dir Model Evaluation for Mode 2                 No
 6     Multi-Dir Model Evaluation for Mode 1                 No
 
 Note: Mode 5 and Mode 6 require the corresponding model from Mode 1 and Mode 2, respectively.
```

#### Data

Put train data sequences inside different folders in Train. Test data goes inside the corresponding Mode folder in Test. 
Sample training data and testing data have been provided for each mode. 

Note: Mode 5 and Mode 6 Test supports multiple folders for different sequences in Mode folder.

#### How to Train

```
python main.py --is_train=True --train_mode = #
```

If you want to see all the flags:
```
python main.py - h
```

#### How to Test

Put test images inside the desired Mode folder in Test folder.
Then run the following command:
```
python main.py --is_train=False --train_mode=#
```

## Tools

The scripts and tools we used for this project are available in Tools.

The table below shows gives a description of each tool/script.

```
Name                                                             Description
PSNR_MultiDir_Calc.py                    Generates PSNR and RGB difference maps for sequences in multiple directories.                       
tubeDownloader.py                        Downloads Youtube videos given URLs in a text file
VideoFrameExtractorPowerTool.py          Extracts sequences from videos
colorBarResultsGenerator.py              Generates grayscale difference maps with colorbars for frames in different sequences.
BulkImageConverter.exe                   Image type conversion tool
```

In addition, Video List folder contains URLs Youtube videos we used to prepare test and training data. 

## PSNR Analysis

PSNR Analysis directory contains scripts and tools used in this project for PSNR analysis.

extractPSNR_test.py and extractPSNR_train.py are used to generate statistics including average PSNR etc. along with mode 1 and mode 2 comparison graphs. 

## Result

A complete collection of the results for Mode 1 and Mode 2 SR is available in this [Google Drive](https://drive.google.com/drive/folders/1sL2Gdc12WQ-lv6pTagdKXpLFlszSkA-Z?usp=sharing). 
















