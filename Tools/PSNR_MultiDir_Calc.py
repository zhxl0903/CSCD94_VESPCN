# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:01:10 2018

@author: HP_OWNER
"""

import numpy as np
import math
import cv2
import glob
import os


def imsave(image, path):
    
    # NOTE: because normial, we need mutlify 255 back    
    cv2.imwrite(path, image*1.0)
    
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

labelPath = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\Test Labels\\fromTrain"
inputDir = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Comparisons\\Mode 1\\fromTrain"
resultDir = "C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\result"
psnrPath = "C:\\Users\\HP_OWNER\Desktop\\TensorFlow-ESPCN\\Tools"

f = open(os.path.join(psnrPath, 'psnr.txt'), 'w')

psnrList = []

# Calculates PSNR for each SR frame in different sequences
# and saves its RGB difference map in a folder with the
# the corresponding sequence name
for root, dirs, files in os.walk(inputDir):
    if dirs != []:
        for folder in dirs:
            
            # Gets testing results
            dataFolderDir = os.path.join(inputDir, folder)        
                    
            # Makes set of all dataset file path
            data = glob.glob(os.path.join(dataFolderDir, "*.png"))
                    
            # Sorts by number in file name
            data.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                       os.path.basename(f)))))
            
            # Prepares labels            
            labelFolderDir = os.path.join(labelPath, folder)
            label = glob.glob(os.path.join(labelFolderDir, "*.png"))
            
            label.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                       os.path.basename(f)))))
													   
            # Creates folder for sequence to save its RGB difference maps
            folderPath = os.path.join(resultDir, folder)
            os.makedirs(folderPath)
            
            print('PSNR for imageSet ' + folder + '\n')
            f.write('PSNR for imageSet ' + folder + '\n')
            f.write('\n')
            
			# Computes PSNR and saves difference map for each frame in sequence 
            psnrSub = []
            for i in range(len(data)):
                hr = cv2.imread(label[i])
                lr = cv2.imread(data[i])
                
                p = psnr(lr, hr, scale = 3)
                psnrSub.append(p)
                
                f.write('PSNR for image ' + str(i) + ': ' + str(p) + '\n')
                print('PSNR for image ' + str(i) + ': ' + str(p))
                
                hr = np.array(hr, dtype=np.float32)
                lr = np.array(lr, dtype=np.float32)
        
                imsave(np.abs(hr-lr), os.path.join(folderPath, 'diffMap'+str(i)+'.png')) 
            print('Average PSNR for image set ' + folder + ' : ' +  str(np.mean(psnrSub)))
            f.write('Average PSNR for image set ' + folder + ' : ' + str(np.mean(psnrSub)) + '\n')
            psnrList.append(np.mean(psnrSub))

print('Average PSNR: ' + str(np.mean(psnrList)))
f.write('Average PSNR: ' + str(np.mean(psnrList)) + '\n')
f.close()
                