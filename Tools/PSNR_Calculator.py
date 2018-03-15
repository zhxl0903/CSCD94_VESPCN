# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:30:40 2018

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

if __name__ == "__main__":
    
    dataFolder = 'v28s0'
    psnrLst = []
    file = open("PSNR.txt", "w")
    for i in range(376):
        data_LR = glob.glob(os.path.join('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Comparison Different Video Categories\\Mode1',  ('result' + str(i) + '.png')))
        data_HR = glob.glob(os.path.join('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Test Data New\\1080p_one_folder', 'test (' + str(i+1) + ').png'))
        
        hr = cv2.imread(data_HR[0])
        lr = cv2.imread(data_LR[0])
        
        p = psnr(lr, hr, scale = 3)
        psnrLst.append(p)
        file.write("Image " + str(i) + " PSNR: " + str(p) + "\n")
        print('PSNR for result image ', i, ' : ', p)
        
        hr = np.array(hr, dtype=np.float32)
        lr = np.array(lr, dtype=np.float32)
        
        imsave(np.abs(hr-lr), 'C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\result\\'+'diffMap'+str(i)+'.png') 
    print('Average PSNR: ', np.mean(psnrLst))
    file.write('Average PSNR: ' + str(np.mean(psnrLst)) + '\n')
    file.close()
    