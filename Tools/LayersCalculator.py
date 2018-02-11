# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:16:10 2018

@author: HP_OWNER
"""

import math

def computeOutDim(i, k, p, s):
    return(math.floor((i - k + 2*p)/s) + 1)

h = 256
w = 144

def getOutputDim(h):
    
    ls = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        
                        cl1h =  computeOutDim(h, 5, i, 2)
                        cl2h = computeOutDim(cl1h, 3, j, 1)
                        cl3h = computeOutDim(cl2h, 3, k, 1)
                        cl4h = computeOutDim(cl3h, 3, l, 1)
                        cl5h = computeOutDim(cl4h, 3, m, 1)
                        
                        if cl5h == h/2:
                            ls.append((i,j,k,l,m))
    return ls

def getOutputDimKernelSearch(h):
    ls = []
    for i in range(1,6,2):
        for j in range(1,6,2):
            for k in range(1,6,2):
                for l in range(1,6,2):
                    for m in range(1,6,2):
                        
                        cl1h =  computeOutDim(h, i, 0, 2)
                        cl2h = computeOutDim(cl1h, j, 0, 1)
                        cl3h = computeOutDim(cl2h, k, 0, 1)
                        cl4h = computeOutDim(cl3h, l, 0, 1)
                        cl5h = computeOutDim(cl4h, m, 0, 1)
                        
                        if cl5h == h/4:
                            ls.append((i,j,k,l,m))
    return ls

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))
    
cl1h =  computeOutDim(h, 5, 0, 2)
cl2h = computeOutDim(cl1h, 3, 0, 1)
cl3h = computeOutDim(cl2h, 5, 0, 2)
cl4h = computeOutDim(cl3h, 3, 2, 1)
cl5h = computeOutDim(cl4h, 3, 2, 1)

cl1w =  computeOutDim(w, 5, 0, 2)
cl2w = computeOutDim(cl1w, 3, 0, 1)
cl3w = computeOutDim(cl2w, 5, 0, 2)
cl4w = computeOutDim(cl3w, 3, 2, 1)
cl5w = computeOutDim(cl4w, 3, 2, 1)

cuph = cl5h * 4 
cupw = cl5w * 4

fl1h =  computeOutDim(cuph, 5, 0, 2)
fl2h = computeOutDim(fl1h, 3, 0, 1)
fl3h = computeOutDim(fl2h, 3, 1, 1)
fl4h = computeOutDim(fl3h, 3, 2, 1)
fl5h = computeOutDim(fl4h, 3, 2, 1)

fl1w =  computeOutDim(cupw, 5, 0, 2)
fl2w = computeOutDim(fl1w, 3, 0, 1)
fl3w = computeOutDim(fl2w, 3, 1, 1)
fl4w = computeOutDim(fl3w, 3, 2, 1)
fl5w = computeOutDim(fl4w, 3, 2, 1)

fuph = fl5h * 2
fupw = fl5w * 2

print ('Course Flow')
print ('Layer1', cl1h, ' ', cl1w)
print ('Layer2', cl2h, ' ', cl2w)
print ('Layer3', cl3h, ' ', cl3w)
print ('Layer4', cl4h, ' ', cl4w)
print ('Layer5', cl5h, ' ', cl5w)
print('x4 Upscale', cuph, ' ', cupw )

print('Fine Flow')
print ('Layer1', fl1h, ' ', fl1w)
print ('Layer2', fl2h, ' ', fl2w)
print ('Layer3', fl3h, ' ', fl3w)
print ('Layer4', fl4h, ' ', fl4w)
print ('Layer5', fl5h, ' ', fl5w)
print('x2 Upscale', fuph, ' ', fupw)
