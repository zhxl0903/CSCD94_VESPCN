# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 19:58:15 2018

@author: HP_OWNER
"""

from pytube import YouTube

yt = YouTube('https://www.youtube.com/watch?v=9Szts88zY4o')
stream = yt.streams.filter(resolution = '144p').all()
print(stream)

for s in stream:
    s.download('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN')
