# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:12:25 2018

@author: HP_OWNER
"""

import numpy as np
import math
#import cv2
import glob
import os
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.image as mpimg

'''def imsave(image, path):
    
    # NOTE: because normial, we need mutlify 255 back    
    cv2.imwrite(path, image*1.0)'''
    
def psnr(target, ref, scale):
	#assume RGB image
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    eps = np.finfo(np.float64).eps
    if(rmse == 0):
        rmse = eps 
    return 20*math.log10(255.0/rmse)


# Defines RGB Difference Map Path and Result Path
DiffMapPath = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Comparisons\\Mode 2\\fromTrainNew\\diffMaps"
resultDir = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\result"

# Generates gray scale difference maps for frames in different sequences
# and saves them to folders with the corresponding sequence names
for root, dirs, files in os.walk(DiffMapPath):
    if dirs != []:
        for folder in dirs:
            
            print('Processing folder: ' + folder)
            # Gets testing results
            dataFolderDir = os.path.join(DiffMapPath, folder)        
                    
            # make set of all dataset file path
            data = glob.glob(os.path.join(dataFolderDir, "*.png"))
                    
            # Sorts by number in file name
            data.sort(key=lambda f: int(''.join(filter(str.isdigit,
			os.path.basename(f)))))
			
            folderPath = os.path.join(resultDir, folder)
            os.makedirs(folderPath)
            
			# Generates grayscale difference map for each frame in sequence
            for i in range(len(data)):
                
                print('Processing image: ' + os.path.basename(data[i]))
                
                fig, ax = plt.subplots()
                img = mpimg.imread(data[i])
				
				# Converts RGB Difference Map to grayscale
                greyScale = np.dot(img[..., :3], [0.299, 0.587, 0.114])

                print(np.shape(greyScale))

                img1 = ax.imshow(greyScale, cmap='gray')
                fig.colorbar(img1, ax=ax)
                
                fig_savePath = os.path.join(folderPath, os.path.basename(data[i]))
                fig.savefig(fig_savePath)
                plt.close(fig)
                plt.clf()

