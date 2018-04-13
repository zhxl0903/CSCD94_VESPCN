# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:13:05 2018

@author: HP_OWNER
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mp

font = {'family': 'normal',
        'size': 20}

mp.rc('font', **font)


l = []
f = open('Mode1_fromTrain.txt', 'r')
l = f.readlines()

p = []
psnr1 = []
for i in range(len(l)):
    if ('PSNR for image ' in l[i] and ('image 0:' not in l[i]) and ('image 4:' not in l[i]) and ('image 24:' not in l[i])) and ('Average PSNR for image ' not in l[i]):
        s = (l[i].strip('\n')).split()
        p.append(float(s[-1]))
        psnr1.append(float(s[-1]))
f.close()
print('Mode 1 PSNR: ' + str(np.mean(psnr1)))

plt.figure(figsize=(25,25))

plt.plot(p, label='Mode 1',  linewidth=3)

'''l=[]
f = open('Mode1_fromTrain.txt' ,'r')
l = f.readlines()

p = []
for i in range(len(l)):
    if('PSNR for image ' in l[i]):
        s = (l[i].strip('\n')).split()
        p.append(float(s[-1]))
f.close()


plt.plot(p)'''

l = []
f = open('Mode2_fromTrain.txt', 'r')
l = f.readlines()

p = []
psnr2 = []

for i in range(len(l)):
    if('PSNR for image ' in l[i]) and ('Average PSNR for image ' not in l[i]):
        s = (l[i].strip('\n')).split()
        p.append(float(s[-1]))
        psnr2.append(float(s[-1]))
f.close()

print('Mode 2 PSNR: ' + str(np.mean(psnr2)))

ps1 = np.array(psnr1)
ps2 = np.array(psnr2)

print('Number of samples in Mode 1 Train Data: ', len(ps1))
print('Number of samples in Mode 2 Train Data: ', len(ps2))
print('Number of samples in Mode 1 with higher PSNR than Mode 2: ', np.sum(ps1 > ps2))
print('Number of samples in Mode 2 with higher PSNR than Mode 1: ', np.sum(ps1 < ps2))

plt.plot(p, label='Mode 2', linewidth=3)

plt.legend()
plt.ylabel('PSNR (dB)')
plt.xlabel('Image')
plt.title('PSNR on Train Data x3 SR')
plt.grid()

plt.savefig('train_PSNR.png')

'''l = []
f = open('Mode2_fromTrain.txt' ,'r')
l = f.readlines()

p = []
for i in range(len(l)):
    if('PSNR for image ' in l[i]):
        s = (l[i].strip('\n')).split()
        p.append(float(s[-1]))
f.close()


plt.plot(p)'''

plt.show()
