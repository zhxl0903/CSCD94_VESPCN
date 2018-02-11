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
    cap = cv2.VideoCapture("Solo A Star Wars Story Big Game TV Spot (45)_144p.mp4")
    
    for i in range(10000):
        ret, frame = cap.read()  
        cv2.imwrite('C:\\zhxl0903\\CSCD94H3\\Optical Flow\\TestData\\frame' + str(i) + '.bmp', frame)
    
    cap.release()
    cv2.destroyAllWindows()