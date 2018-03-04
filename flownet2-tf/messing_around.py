import cv2
import glob
from src.net import Mode
from src.flownet2.flownet2 import FlowNet2
from scipy.misc import imread, imshow, imsave
import numpy as np
import tensorflow as tf

def test(input_a, input_b):
    # Create a new network
    net = FlowNet2(mode=Mode.TEST)

    # Train on the data
    return net.run(checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
                    input_a=input_a,
                    input_b=input_b)

        
video_path = glob.glob('/home/kumar/Downloads/*Deadpool*.mp4')
vid = cv2.VideoCapture(video_path[0])

fourcc = cv2.VideoWriter_fourcc(*'DIVX')



#------------------------------------------------------------------------------------
ret1, frame1 = vid.read()
ret2, frame2 = vid.read()

a = 1
b = 1

size_x = (np.size(frame1,0) // a) * a
size_y = (np.size(frame1,1) // b) * b

result = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))

frame1 = frame1[0:size_x, 0:size_y,:]
frame2 = frame2[0:size_x, 0:size_y,:]


print(frame1.shape)
# # print(frame2)
# print('------------------------\n')
# f1 = cv2.imread('/home/kumar/project/flownet2-tf/data/samples/1img0.ppm')
# print(f1.shape)

# f2 = cv2.imread('/home/kumar/project/flownet2-tf/data/samples/1img1.ppm')
# # print(f2)

out = test(frame1,frame2)
print("hello-----------------------------------------------------------\n")

result.write(out)

tf.reset_default_graph()
# print(out.shape)
# out = test(frame1, frame2)

cv2.imshow('raaaaa',out)
cv2.waitKey(1)







for i in range(0,50):

    ret1, frame1 = vid.read()
    ret2, frame2 = vid.read()

    frame1 = frame1[0:size_x, 0:size_y,:]
    frame2 = frame2[0:size_x, 0:size_y,:]
    
    # print(frame1.shape)
    # # print(frame2)
    # print('------------------------\n')
    # f1 = cv2.imread('/home/kumar/project/flownet2-tf/data/samples/1img0.ppm')
    # print(f1.shape)

    # f2 = cv2.imread('/home/kumar/project/flownet2-tf/data/samples/1img1.ppm')
    # # print(f2)

    out = test(frame1,frame2)
    print("hello-----------------------------------------------------------\n")

    result.write(out)
    tf.reset_default_graph()
    # print(out.shape)
    # out = test(frame1, frame2)

    cv2.imshow('raaaaa',out)
    # cv2.imshow(out)
    cv2.waitKey(1)

vid.release()
result.release()