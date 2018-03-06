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

if __name__=='__main__':
    
    for i in range(30):
        
        print('Working on video ', i, ' ...')
        cap = cv2.VideoCapture("v" + str(i) + ".mp4")
    
        try:
            for j in range(3000):
                ret, frame = cap.read()  
                cv2.imwrite('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\360p\\v' + str(i) + '\\' + str(j) + '.png', frame)
        except:
            print('We have less than 3000 frames in video')
        cap.release()
        cv2.destroyAllWindows()