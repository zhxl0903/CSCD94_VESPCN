import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mp

font = {'family': 'normal',
        'size': 20}

mp.rc('font', **font)

l = []
f = open('Mode1_fromTest.txt', 'r')
l = f.readlines()

psnr1 = []
psnr2 = []

# Computes PSNRs for Mode 1 data
p = []
for i in range(len(l)):
    if ('PSNR for image ' in l[i] and ('image 0:' not in l[i])
       and ('image 4:' not in l[i])) and ('Average PSNR for image ' not in l[i]):
        s = (l[i].strip('\n')).split()
        p.append(float(s[-1]))
        psnr1.append(float(s[-1]))
f.close()

plt.figure(figsize=(25, 25))
plt.plot(p, label='Mode 1',  linewidth=3)

print('Mode 1 PSNR: ' + str(np.mean(psnr1)))

l = []
f = open('Mode2_fromTest.txt', 'r')
l = f.readlines()

# Computes PSNRs for Mode 2 data
p = []
for i in range(len(l)):
    if('PSNR for image ' in l[i]) and ('Average PSNR for image ' not in l[i]):
        s = (l[i].strip('\n')).split()
        p.append(float(s[-1]))
        psnr2.append(float(s[-1]))
f.close()

plt.plot(p, label='Mode 2', linewidth=3)

print('Mode 2 PSNR: ' + str(np.mean(psnr2)))

ps1 = np.array(psnr1)
ps2 = np.array(psnr2)

print('Number of samples in Mode 1 Test Data: ', len(ps1))
print('Number of samples in Mode 2 Test Data: ', len(ps2))
print('Number of samples in Mode 1 with higher PSNR than Mode 2: ', np.sum(ps1 > ps2))
print('Number of samples in Mode 2 with higher PSNR than Mode 1: ', np.sum(ps1 < ps2))

plt.legend()
plt.ylabel('PSNR (dB)')
plt.xlabel('Image')
plt.title('PSNR on Testing Data x3 SR')
plt.grid()

# Saves PSNR graph 
plt.savefig('test_PSNR.png')
plt.show()
