# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 19:58:15 2018

@author: HP_OWNER
"""

from pytube import YouTube

# Trailer 1: https://www.youtube.com/watch?v=dNW0B0HsvVs
# Trailer 2: https://www.youtube.com/watch?v=Q0CbN8sfihY

# Pytube documentation


file = open("videos.txt", "r")
paths = file.readlines()
file.close()

for i in range(50):
    try:
        yt = YouTube(paths[i].strip('\n'))
        stream = yt.streams.filter(resolution = '1080p', file_extension='mp4').all()
        print('Downloading video ', i, ' 1080p in list.')
        for s in stream:
            s.download('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\New\\1080p')
        
        yt2 = YouTube(paths[i].strip('\n'))
        stream2 = yt2.streams.filter(resolution = '360p', file_extension='mp4').all()
        print('Downloading video ', i, ' 360p in list.')
        for s in stream2:
            s.download('C:\\Users\\HP_OWNER\\Desktop\\TensorFlow-ESPCN\\Videos\\New\\360p')
    except:
        print('Failed to download video ', i, ' in list.')


