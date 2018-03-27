# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 08:28:30 2018

@author: HP_OWNER
"""

import tensorflow as tf
import numpy as np
import math
import time
import os
import glob
import cv2
import scipy as sp
import shutil as sh
from distutils.dir_util import copy_tree

#os.system("conda activate tensorflow")
#os.system("cd C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN")
#os.system("python main.py --train_mode=2 --is_train=False")

'''
Removes all files in directory d.
'''
def removeFiles (d):
    files = glob.glob(os.path.join(d, "*"))
    for f in files:
        print(f)
        os.remove(f)

#removeFiles("C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\result\\*")
#copy_tree("C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Train\\v0s0", "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\result\\cool")
        
os.system("conda activate tensorflow")
os.system("cd C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN")

mainPath = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN"
trainPath = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Train"
testMode2 = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Test\\Mode2" 
testMode1 = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Test\\Mode1" 
compMode2 = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Comparisons\\Mode 2"
resultPath = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\result"

taskList = glob.glob(os.path.join(trainPath, "*"))
for folder in taskList:
    print('Processing images in folder: ' + os.path.basename(folder))
    print('Removing test mode directory files...')
    removeFiles(testMode2)
    print('Copying new files to test mode directory...')
    copy_tree(folder, testMode2)
    print('Removing files from result directory...')
    removeFiles(resultPath)
    print('Running testing on files from ' + os.path.basename(folder) + ' ...')
    os.system("python main.py --train_mode=2 --is_train=False")
    
    print('Copying results to comparison folder...')
    comparisonDir = os.path.join(compMode2, os.path.basename(folder))
    os.makedirs(comparisonDir)
    copy_tree(resultPath, comparisonDir)
    
    removeFiles(resultPath)
    
    
    