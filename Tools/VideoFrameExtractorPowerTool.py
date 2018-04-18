# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:50:48 2018

@author: HP_OWNER
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
import time
import os 
import glob
import random as rand

if __name__=='__main__':
    
    
    rand.seed()
    
	# Directory containing 360p videos
    data_dir1 = os.path.join('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\New',
                            '360p')
            
    # make set of all dataset file path for 360p
    data1 = glob.glob(os.path.join(data_dir1, "*.mp4"))

    # Sorts by number in file name
    data1.sort(key=lambda f: os.path.basename(f))
    
	# Directory containing the corresponding 1080p videos
    data_dir2 = os.path.join('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\New',
                            '1080p')
            
    # make set of all dataset file path for 360p
    data2 = glob.glob(os.path.join(data_dir2, "*.mp4"))

    # Sorts by number in file name
    data2.sort(key=lambda f: os.path.basename(f))
    
    numFiles = len(data1)
    
    for i in range(numFiles):
        
        print('Working on video ', str(i))
        
        # Generates random frame capture points for each interval
        startList = []
        for l in range(15):
            startList.append(rand.randint(l*200, l*200 + 195))
        
        cap = cv2.VideoCapture(data1[i])
        cap2 = cv2.VideoCapture(data2[i])
        
        k = -1
        for j in range(3000):
            
            # Switches indices for selection from different intervals
            if (j % 200 == 0):
                k = k + 1
                
            if(j % 200 == 0):
                
                # Computes paths for the folders for video i 360p and 1080p
                path = os.path.join('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\New_out\\360p', os.path.basename(data1[i]).strip('.mp4') + '_' + str(k))
                path1 = os.path.join('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\New_out\\1080p', os.path.basename(data2[i]).strip('.mp4') + '_' + str(k))
                
                os.makedirs(path)
                os.makedirs(path1)
                
            try:
                
                # Reads video frames
                ret, frame = cap.read()
                ret1, frame1 = cap2.read()
                    
                if (j >= startList[k] and j < startList[k] + 5):
                    
                    print('Writing frame ' + str(j))
                    cv2.imwrite(os.path.join(path, str(j)+'.png'), frame)
                    cv2.imwrite(os.path.join(path1, str(j)+'.png'), frame1)
            except Exception as e:
                print('End of frames reached for video ' + str(i))
                print(e)
                break
    
        cap.release()
        cap2.release()
    cv2.destroyAllWindows()