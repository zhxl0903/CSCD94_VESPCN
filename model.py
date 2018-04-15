import tensorflow as tf
import numpy as np
import math
import time
import os
import glob
import cv2
import scipy as sp

from utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    checkimage,
    imsave,
    imread,
    load_data,
    preprocess,
    modcrop
)
from PSNR import psnr


class ESPCN(object):
    """
        This class serves as the manager of all activities
        including model initialisation, training, testing, model saving,
        and model loading. An object of this class type is created by main.py
        and train method can then be called to start training iff FLAGS.is_train
        = True.
    """

    def __init__(self,
                 sess,
                 image_size,
                 is_train,
                 train_mode,
                 scale,
                 batch_size,
                 c_dim,
                 load_existing_data,
                 config
                 ):

        """
               This method is constructor for class ESPSCN.

               Inputs:
                    sess: Session object for this model
                    image_size: size of image for training
                    is_train: True iff training
                    train_mode: 0 is spatial transformer only
                                1 is single frame 9-Layer ESPCN
                                2 is 9-Layer-Early-Fusion VESPCN with MC
                                3 is Bicubic (No Training Required)
                                4 is SRCNN
                                5 is Multi-Dir-Output Mode 2
                                6 is Multi-Dir-Output Mode 1
                    scale: upscaling ratio for super resolution
                    batch_size: batch size for training
                    c_dim: number of channels of each input image
                    config: config object

               Returns: None
        """

        # Initializes layer memory dictionary
        self.layerOutputs = dict()
        self.load_existing_data = load_existing_data
        
        self.sess = sess
        self.image_size = image_size
        self.is_train = is_train
        self.c_dim = c_dim
        self.scale = scale
        self.train_mode = train_mode
        self.batch_size = batch_size
        self.config = config
        self.build_model()

    def build_model(self):

        """
           This method builds network based on training mode. Placeholders are
           setup based on training mode and is_train. Predictions are obtained
           by calling method model. Saver is initialized and loss functions are
           initialized based on training mode.

           Given: None
           Returns: None
        """

        if self.is_train:
            
            # Prepares placeholders based on train_mode used for model training
            if self.train_mode == 0:
                
                # Sets up placeholders for training spatial transformer 
                # if train mode is 0
                self.images_curr_prev = tf.placeholder(tf.float32,
                                                       [None, self.image_size,
                                                        self.image_size,
                                                        2*self.c_dim],
                                                       name='images_curr_prev')
                self.labels = tf.placeholder(tf.float32,
                                             [None, self.image_size,
                                              self.image_size,
                                              self.c_dim], name='labels')
            elif self.train_mode == 1 or self.train_mode == 6:
                
                # Sets up placeholders for training subpixel net 
                # if train mode is 1
                self.images_in = tf.placeholder(tf.float32, [None,
                                                             self.image_size,
                                                             self.image_size,
                                                             self.c_dim],
                                                name='image_in')
                self.labels = tf.placeholder(tf.float32, 
                                             [None,
                                              self.image_size * self.scale,
                                              self.image_size * self.scale,
                                              self.c_dim], name='labels')
            elif self.train_mode == 4:
                self.images_in = tf.placeholder(tf.float32, [None,
                                                             self.image_size
                                                             * self.scale,
                                                             self.image_size
                                                             * self.scale,
                                                             self.c_dim],
                                                name='image_in')
                self.labels = tf.placeholder(tf.float32, 
                                             [None,
                                              self.image_size * self.scale,
                                              self.image_size * self.scale,
                                              self.c_dim], name='labels')
            elif self.train_mode == 5:

                # Prepares placeholders for mode 2 with multiple test folders
                self.images_prev_curr = tf.placeholder(tf.float32,
                                                       [None, 
                                                        self.image_size,
                                                        self.image_size,
                                                        2*self.c_dim],
                                                       name='images_prev_curr')
                self.images_next_curr = tf.placeholder(tf.float32,
                                                       [None,
                                                        self.image_size,
                                                        self.image_size,
                                                        2*self.c_dim],
                                                       name='images_next_curr')
                self.labels = tf.placeholder(tf.float32,
                                             [None,
                                              self.image_size * self.scale,
                                              self.image_size * self.scale,
                                              self.c_dim], name='labels')
                
            else:
                
                # Sets up placeholders for training full network 
                # if train mode is 2 or other numbers
                self.images_prev_curr = tf.placeholder(tf.float32,
                                                       [None, 
                                                        self.image_size,
                                                        self.image_size,
                                                        2*self.c_dim],
                                                       name='images_prev_curr')
                self.images_next_curr = tf.placeholder(tf.float32,
                                                       [None,
                                                        self.image_size,
                                                        self.image_size,
                                                        2*self.c_dim],
                                                       name='images_next_curr')
                self.labels = tf.placeholder(tf.float32,
                                             [None,
                                              self.image_size * self.scale,
                                              self.image_size * self.scale,
                                              self.c_dim], name='labels')
        else:

            # Computes shape of placeholder using image loaded from sample image
            print('Train Mode:', self.train_mode)
            data = load_data(self.is_train, self.train_mode)
            input_ = imread(data[0][0])       
            self.h, self.w, c = input_.shape
            
            # Prepares placeholders for model evaluation based on traini_mode used
            if self.train_mode == 0:
                
                # Sets up placeholders for training spatial transformer
                # if train mode is 0
                self.images_curr_prev = tf.placeholder(tf.float32,
                                                       [None,
                                                        self.h,
                                                        self.w,
                                                        2*self.c_dim],
                                                       name='images_curr_prev')
                self.labels = tf.placeholder(tf.float32,
                                             [None,
                                              self.h,
                                              self.w,
                                              self.c_dim], name='labels')
            elif self.train_mode == 1 or self.train_mode == 3 or self.train_mode == 6:

                # Sets up placeholders for training subpixel net
                # if train mode is 1
                self.images_in = tf.placeholder(tf.float32,
                                                [None,
                                                 self.h,
                                                 self.w,
                                                 self.c_dim], name='images_in')
                self.labels = tf.placeholder(tf.float32,
                                             [None,
                                              self.h * self.scale,
                                              self.w * self.scale,
                                              self.c_dim], name='labels')
            elif self.train_mode == 4:
                
                # Sets up placeholders for training subpixel net 
                # if train mode is 1
                self.images_in = tf.placeholder(tf.float32,
                                                [None,
                                                 self.h * self.scale,
                                                 self.w * self.scale,
                                                 self.c_dim], name='images_in')
                self.labels = tf.placeholder(tf.float32,
                                             [None,
                                              self.h * self.scale,
                                              self.w * self.scale,
                                              self.c_dim], name='labels')
            else:
                
                # Sets up placeholders for training full network 
                # if train mode is 2 or other numbers
                self.images_prev_curr = tf.placeholder(tf.float32,
                                                       [None,
                                                        self.h,
                                                        self.w,
                                                        2*self.c_dim],
                                                       name='images_prev_curr')

                self.images_next_curr = tf.placeholder(tf.float32,
                                                       [None,
                                                        self.h,
                                                        self.w,
                                                        2*self.c_dim],
                                                       name='images_next_curr')
                self.labels = tf.placeholder(tf.float32,
                                             [None,
                                              self.h * self.scale,
                                              self.w * self.scale,
                                              self.c_dim], name='labels')
        
        if self.train_mode == 0 or self.train_mode == 1 or self.train_mode == 3 or self.train_mode == 4 or \
                self.train_mode == 6:
            self.pred = self.model()
        else:
            self.pred, self.imgPrev, self.imgNext = self.model()
        
        # Prepares loss function based on training mode
        if self.train_mode == 0:
            
            # Defines loss function for training a single spatial transformer
            self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        elif self.train_mode == 1 or self.train_mode == 4 or self.train_mode == 6:
            
            # Defines loss function for training subpixel convnet
            self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
            print('Mode 1/4/6: Mean-Squared Loss Activated')
        elif self.train_mode == 2 or self.train_mode == 5:
            
            # Defines loss function for Spatial Transformer and VESPCN joint training
            self.loss = tf.reduce_mean(tf.square(self.labels - self.pred)) \
                      + 0.01*tf.reduce_mean(tf.square(self.imgPrev -
                                            self.images_prev_curr[:, :,
                                                                  :,
                                                                  0:self.c_dim])) \
                      + 0.01*tf.reduce_mean(tf.square(self.imgNext -
                                            self.images_prev_curr[:, :, :,
                                                                  0:self.c_dim]))
            
        if self.train_mode != 3:

            # To save checkpoint
            self.saver = tf.train.Saver()

    def spatial_transformer(self, frameSet, reuse=False):

        """
            This method constructs spatial transformer given frameSet.

            Inputs:
            frameSet: frameSet containing 2 images: current frame and neighbouring frame
                      Tensor of dimension: [nBatch, imgH, imgW, 6]

            reuse: True iff Weights are the same as network where reuse is false

            Returns: Output image of this spatial transformer network
                     Tensor of dimension: [nBatch, imgH, imgW, 3]
        """

        # Zero initialization
        biasInitializer = tf.zeros_initializer()
        
        # Orthogonal initialization with gain sqrt(2)
        weight_init = tf.orthogonal_initializer(np.sqrt(2))
        
        # Course flow
        t1_course_l1 = tf.layers.conv2d(frameSet,  24, 5, padding='same',
                                        strides=(2, 2),
                                        activation=tf.nn.relu,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l1', reuse=reuse)

        t1_course_l2 = tf.layers.conv2d(t1_course_l1,  24, 3, padding='same',
                                        strides=(1, 1),
                                        activation=tf.nn.relu,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l2', reuse=reuse)
        t1_course_l3 = tf.layers.conv2d(t1_course_l2,  24, 5, padding='same',
                                        strides=(2, 2),
                                        activation=tf.nn.relu,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l3', reuse=reuse)
        t1_course_l4 = tf.layers.conv2d(t1_course_l3,  24, 3, padding='same',
                                        strides=(1, 1),
                                        activation=tf.nn.relu,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l4', reuse=reuse)
        t1_course_l5 = tf.layers.conv2d(t1_course_l4,  32, 3, padding='same',
                                        strides=(1, 1),
                                        activation=tf.nn.tanh,
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_course_l5', reuse=reuse)
        
        # Defines Course Flow Output
        # Output shape: (-1, l, w, 2)
        t1_course_out = self.PS2(t1_course_l5, 4, 2)
        
        if not reuse:
            self.layerOutputs['courseFlow'] = t1_course_out
        
        # Course Warping
        # Gets target image to be warped
        targetImg = frameSet[:, :, :, self.c_dim:self.c_dim*2]
        
        # Generates tensor of dimension [-1, h, w, 3+2]
        t1_course_warp_in = tf.concat([targetImg, t1_course_out], 3)
        
        # Applies warping using 3D convolution to estimate image at time
        # t=t
        # Kernel size 3 is used to apply flow to image based on neighbouring
        # flows of pixel
        t1_course_warp = tf.layers.conv2d(t1_course_warp_in, 3, 3,
                                          padding='same',
                                          activation=tf.nn.tanh,
                                          kernel_initializer=weight_init,
                                          bias_initializer=biasInitializer,
                                          name='t1_course_warp', reuse=reuse)
        
        # Fine flow 
        # Stacks images input, course flow estimation, 
        # and course flow warped image along dimension 3
        t1_fine_in = tf.concat([frameSet, t1_course_warp,
                                t1_course_out], 3)
        
        t1_fine_l1 = tf.layers.conv2d(t1_fine_in,  24, 5, padding='same',
                                      strides=(2, 2),
                                      activation=tf.nn.relu,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l1', reuse=reuse)
        
        t1_fine_l2 = tf.layers.conv2d(t1_fine_l1,  24, 3, padding='same',
                                      strides=(1, 1),
                                      activation=tf.nn.relu,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l2', reuse=reuse)
        
        t1_fine_l3 = tf.layers.conv2d(t1_fine_l2,  24, 3, padding='same',
                                      strides=(1, 1),
                                      activation=tf.nn.relu,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l3', reuse=reuse)
        
        t1_fine_l4 = tf.layers.conv2d(t1_fine_l3,  24, 3, padding='same',
                                      strides=(1, 1),
                                      activation=tf.nn.relu,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l4', reuse=reuse)
        
        t1_fine_l5 = tf.layers.conv2d(t1_fine_l4,  8, 3, padding='same',
                                      strides=(1, 1),
                                      activation=tf.nn.tanh,
                                      kernel_initializer=weight_init,
                                      bias_initializer=biasInitializer,
                                      name='t1_fine_l5', reuse=reuse)
        
        # Output shape(-1, l, w, 2)
        t1_fine_out = self.PS2(t1_fine_l5, 2, 2)
        
        if not reuse:
            self.layerOutputs['fineFlow'] = t1_fine_out
        
        # Combines fine flow and course flow estimates
        # Output shape(-1, l, w, 2)
        t1_combined_flow = t1_course_out + t1_fine_out
        
        # Fine Warping
        # Concatenates target image and combined flow channels
        t1_fine_warp_in = tf.concat([targetImg, t1_combined_flow], 3)
        
        # Applies warping using 2D convolution layer to estimate image at time
        # t=t
        # Kernel size 3 is used to apply flow based on neighbouring flows of
        # pixel
        # Output shape: (batchSize, h, w, c_dim)
        t1_fine_warp = tf.layers.conv2d(t1_fine_warp_in, 3, 3, padding='same',
                                        activation=tf.nn.tanh, 
                                        kernel_initializer=weight_init,
                                        bias_initializer=biasInitializer,
                                        name='t1_fine_warp', reuse=reuse)

        # Output shape: (batchSize, h, w, c_dim)
        return t1_fine_warp

    def model(self):

        """
            This method generates a network model given self.train_mode
            train mode = 0: model for 1 spatial transformer
            train mode = 1: model for single frame ESPSCN
            train mode = 2: model for VESPCN with 2 spatial transformers taking 2
                            images each
            train mode = 3: model for Bicubic (no training required)
            train mode = 4: model for SRCNN
            train mode = 5: model for multi-dir mode 2 (no training required)
            train mode = 6: model for multi-dir mode 1 (no training required)

            Returns: Output of network if train mode is 0 or 1
                     Output of network and output of 2 spatial transformers of network
                     if train mode is 2
        """

        # Generates motion compensated images from previous and next images
        # using 2 spatial transformers
           
        # Initializes spatial transformer if training mode is 0 or 2
        if self.train_mode == 2 or self.train_mode == 5:

            # Obtains outputs from motion compensated previous and next
            # images which are stacked with the target image for Early Fusion
            imgPrev = self.spatial_transformer(self.images_prev_curr,
                                               reuse=False)
            imgNext = self.spatial_transformer(self.images_next_curr,
                                               reuse=True)
               
            targetImg = self.images_prev_curr[:, :, :, 0:self.c_dim]
            imgSet = tf.concat([imgPrev, targetImg, imgNext], 3)
        elif self.train_mode == 0:
            motionCompensatedImgOut = self.spatial_transformer(self.images_curr_prev, reuse=False)
        else:
            imgSet = self.images_in
       
        wInitializer1 = tf.orthogonal_initializer(np.sqrt(2))
        wInitializer2 = tf.orthogonal_initializer(np.sqrt(2))
        wInitializer3 = tf.orthogonal_initializer(np.sqrt(2))
       
        biasInitializer = tf.zeros_initializer()

        if self.train_mode == 2 or self.train_mode == 5:

            # Connects early fusion network with spatial transformer
            # and subpixel convnet. For collapsing to temporal depth of 1,
            # number of channels produced is 24 by VESPCN paper
            EarlyFusion = tf.layers.conv2d(imgSet,  24, 3, padding='same',
                                           activation=tf.nn.relu,
                                           kernel_initializer=wInitializer1,
                                           bias_initializer=biasInitializer,
                                           name='EF1')

            subPixelIn = EarlyFusion

        elif self.train_mode == 1 or self.train_mode == 6:

            # Connects subpixel convnet to placeholder for feeding single frames
            subPixelIn = tf.layers.conv2d(imgSet,  24, 3, padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=wInitializer1,
                                          bias_initializer=biasInitializer,
                                          name='subPixelIn')
        elif self.train_mode == 3:

            # Sets height and width for resizing based on is_train
            if self.is_train:

                # Sets to training data patch size
                height = self.image_size
                width = self.image_size
            else:

                # Sets to image size
                height = self.h
                width = self.w

            biCubic = tf.image.resize_images(imgSet, [height*self.scale,
                                             width*self.scale],
                                             method=tf.image.ResizeMethod.BICUBIC)

        # Builds Subpixel Network if train mode is 1 or 2
        if self.train_mode == 1 or self.train_mode == 2 or self.train_mode == 5 \
           or self.train_mode == 6:

            conv1 = tf.layers.conv2d(subPixelIn,  24, 3, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=wInitializer1,
                                     bias_initializer=biasInitializer,
                                     name='subPixelL1')
            conv2 = tf.layers.conv2d(conv1,  24, 3, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=wInitializer1,
                                     bias_initializer=biasInitializer,
                                     name='subPixelL2')
            conv3 = tf.layers.conv2d(conv2,  24, 3, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=wInitializer1,
                                     bias_initializer=biasInitializer,
                                     name='subPixelL3')
            conv4 = tf.layers.conv2d(conv3,  24, 3, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=wInitializer1,
                                     bias_initializer=biasInitializer,
                                     name='subPixelL4')
            conv5 = tf.layers.conv2d(conv4,  24, 3, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=wInitializer2,
                                     bias_initializer=biasInitializer,
                                     name='subPixelL5')
            conv6 = tf.layers.conv2d(conv5,  24, 3, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=wInitializer2,
                                     bias_initializer=biasInitializer,
                                     name='subPixelL6')
            conv7 = tf.layers.conv2d(conv6,  24, 3, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=wInitializer2,
                                     bias_initializer=biasInitializer,
                                     name='subPixelL7')

            conv8 = tf.layers.conv2d(conv7,
                                     self.c_dim * self.scale * self.scale,
                                     3, padding='same', activation=None,
                                     kernel_initializer=wInitializer3,
                                     bias_initializer=biasInitializer,
                                     name='subPixelL8')

            ps = self.PS(conv8, self.scale)
        elif self.train_mode == 4:

            # Builds SRCNN network if train_mode is 4
            conv1 = tf.layers.conv2d(imgSet,  64, 9, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=wInitializer1,
                                     bias_initializer=biasInitializer,
                                     name='SRCNN1')
            conv2 = tf.layers.conv2d(conv1,  32, 1, padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=wInitializer1,
                                     bias_initializer=biasInitializer,
                                     name='SRCNN2')
            conv3 = tf.layers.conv2d(conv2,  self.c_dim, 5, padding='same',
                                     activation=None,
                                     kernel_initializer=wInitializer1,
                                     bias_initializer=biasInitializer,
                                     name='SRCNN3')

        # Returns network output given self.train_mode
        if self.train_mode == 0:
            return motionCompensatedImgOut
        elif self.train_mode == 1 or self.train_mode == 6:
            return tf.nn.tanh(ps)
        elif self.train_mode == 3:
            return biCubic
        elif self.train_mode == 4:
            return conv3
        else:
            return tf.nn.tanh(ps), imgPrev, imgNext

    @staticmethod
    def _phase_shift(I, r):

        """
            This method serves as a helper method for PS ad PS2 in the case of
            processing training images. Input tensor X in PS and PS2 has r*r*cdim
            channels which is split into cdim tensors each with r*r channels which
            are processed one at a time by calling this method.

            Note: batch size can be more than 1 in this method

            Input:
                I: Tensor of dimension (batch_size, a, b, r*r)
                r: upscaling ratio
            Returns:
                Tensor of dimension (batch_size, a*r, b*r, 1)
        """

        # NOTE: train with batch size
        
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (-1, a, b, r, r))
        
        # a, [bsize, b, r, r]
        X = tf.split(X, a, 1)  
        
        # bsize, b, a*r, r
        X = tf.concat([tf.squeeze(x) for x in X], 2)  
        
        # b, [bsize, a*r, r]
        X = tf.split(X, b, 1)  
        
        # bsize, a*r, b*r
        X = tf.concat([tf.squeeze(x) for x in X], 2)  
        return tf.reshape(X, (-1, a*r, b*r, 1))

    @staticmethod
    def _phase_shift_test(I, r):

        """
        This method serves as a helper method for PS ad PS2 in the case of
        processing test images. Input tensor X in PS and PS2 has r*r*cdim
        channels which is split into cdim tensors each with r*r channels which
        are processed one at a time by calling this method.

        Note: only single image batches are supported during testing

        Input:
            I: Tensor of dimension (1, a, b, r*r)
            r: upscaling ratio
        Returns:
            Tensor of dimesnion (1, a*r, b*r, 1)
        """

        # NOTE:test without batchsize
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        
        # a, [bsize, b, r, r]
        X = tf.split(X, a, 1)  
        
        # bsize, b, a*r, r
        X = tf.concat([tf.squeeze(x) for x in X], 1)  
        
        # b, [bsize, a*r, r]
        X = tf.split(X, b, 0)  
        
        # bsize, a*r, b*r
        X = tf.concat([tf.squeeze(x) for x in X], 1)  
        return tf.reshape(X, (1, a*r, b*r, 1))

    def PS(self, X, r):

        """
           Performs phase shift operation for tensor of dimension
           (batch_size, img_height, img_width, 3*r*r)

           Inputs:
           X: tensor of dimension (batch_size, img_height, img_width, 3*r*r)
           r: upscaling factor
           c_dim: c_dim of X

           Returns:
           Tensor of shape (batchs_size, img_height*r, img_width*r, 3)
        """

        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            
            # Does concat RGB
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) 
        else:
            
            # Does concat RGB
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) 
        return X

    def PS2(self, X, r, c_dim):

        """
           Performs phase shift operation for tensor of dimension
           (batch_size, img_height, img_width, c_dim)

           Inputs:
           X: tensor of dimension (batch_size, img_height, img_width, c_dim*r*r)
           r: upscaling factor
           c_dim: c_dim of X

           Returns:
           Tensor of shape (batchs_size, img_height*r, img_width*r, c_dim)
        """

        # Main OP that you can arbitrarily use in you tensorflow code
        
        # Evenly splits Xc into c_dim parts along axis 3 (# of channels)
        Xc = tf.split(X, c_dim, 3)
        if self.is_train:
            
            # Does the concat RGB
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) 
        else:

            # Does the concat RGB
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3)
        return X

    def train(self, config):

        """
            This method performs training / testing operations given config.
            Training / Testing is supported for mode 0,1,2,4 from self.train_mode.

            See class __init__() for additional a description of training modes.

            Input:
            config: config object of this class

        """
        # NOTE : if train, the nx, ny are ingnored
        
        # Prepares data if load_existing_data is False
        if not self.load_existing_data:
            input_setup(config)
        
        # Loads data from data_dir
        print('Loading data...')
        data_dir = checkpoint_dir(config)
        input_, label_, paths_= read_data(data_dir, config)
        
        if self.train_mode != 5 and self.train_mode != 6:
            print('Input and Label Shapes:')
            print(input_.shape, label_.shape)
        
        #  Sets optimizer to be Adam optimizer
        if self.train_mode != 3:
            self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())

        counter = 0
        time_ = time.time()
        
        if self.train_mode != 3:
            self.load(config.checkpoint_dir)
        
        # Train
        if config.is_train:
            
            # Shuffles training data
            print('Shuffling data...')
            numData = np.arange(input_.shape[0])
            np.random.shuffle(numData)
            input_ = input_[numData]
            label_ = label_[numData]
            
            # Prepares frame sets for feeding into different spatial
            # transformers if training mode is 2
            if self.train_mode == 2:
                print("Preparing frames sets for spatial transformers...")
                
                curr_prev_imgs = input_[:, :, :, 0:2*self.c_dim]
                curr_next_imgs = np.concatenate((input_[:, :, :,
                                                        0:self.c_dim],
                                                 input_[:, :, :, 2*self.c_dim:3*self.c_dim]), axis=3)
            elif self.train_mode == 4:
                
                # Upscales input data using bicubic interpolation
                print('Upscaling training data using Bicubic Interpolation...')
                
                input_new = []
                for i in range(len(input_)):
                    input_new.append(sp.misc.imresize(input_[i], (self.image_size*self.scale,
                                                                  self.image_size*self.scale), interp='bicubic'))
                input_ = np.array(input_new)
                
            print("Now Start Training...")
            for ep in range(config.epoch):
                
                # Runs by batch images
                batch_idxs = len(input_) // config.batch_size
                final_batch_size = len(input_) % config.batch_size
                
                # increases total number of mini batches by 1 if we have 
                # a reminder batch
                if final_batch_size != 0:
                    batch_idxs = batch_idxs + 1 
                
                for idx in range(0, batch_idxs):
                    
                    # Processes reminder batch if final_batch_size is non-zero
                    # and idx = batch_idxs - 1
                    if (final_batch_size != 0) and (idx == batch_idxs - 1):
                        iterationBatchSize = final_batch_size
                    else:
                        iterationBatchSize = config.batch_size
                    
                    # Prepares images for current batch if train_mode is not 2
                    # Data will be prepared differently for train_mode=2
                    if not (self.train_mode == 2):
                        batch_images = input_[idx * config.batch_size
                                              : idx * config.batch_size
                                              + iterationBatchSize]
                        
                    # Prepares labels for current batch
                    batch_labels = label_[idx * config.batch_size
                                          : idx * config.batch_size
                                          + iterationBatchSize]
                    
                    # print('Batch Shape:', np.shape(batch_images))
                    # print('Label Shape:', np.shape(batch_labels))
                    
                    counter += 1
                    
                    # Feeds data into placeholders based on self.training_mode
                    if config.train_mode == 0:
                        _, err = self.sess.run([self.train_op, self.loss],
                                               feed_dict=
                                               {self.images_curr_prev:
                                                batch_images,
                                                self.labels: batch_labels})
                    elif config.train_mode == 1:
                        _, err = self.sess.run([self.train_op, self.loss],
                                               feed_dict={self.images_in:
                                                          batch_images,
                                                          self.labels:
                                                          batch_labels})
                    elif config.train_mode == 2:
                        
                        # Obtains input frame sets for current batch
                        curr_prev = curr_prev_imgs[idx * config.batch_size
                                                   : idx * config.batch_size + iterationBatchSize]
                        curr_next = curr_next_imgs[idx * config.batch_size:idx * config.batch_size
                                                   + iterationBatchSize]
                        
                        # Feeds images and labels for current batch
                        _, err = self.sess.run([self.train_op, self.loss],
                                               feed_dict=
                                               {self.images_prev_curr: curr_prev,
                                                self.images_next_curr: curr_next,
                                                self.labels: batch_labels})
                    elif config.train_mode == 4:

                        _, err = self.sess.run([self.train_op, self.loss],
                                               feed_dict={self.images_in:
                                                          batch_images,
                                                          self.labels:
                                                          batch_labels})
                        
                    if counter % 10 == 0:
                        print("Epoch: ", (ep+1), " Step: ", counter,
                              " Time: ",
                              (time.time()-time_), " Loss: ", err)
                        
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
        # Test
        else:
            print("Now Start Testing...")
            
            psnrLst = []
            errorLst = []
            
            if self.train_mode == 0:
                
                # Obtains course flow tensor
                courseFlow = self.layerOutputs['courseFlow']
            
                # Obtains fine flow tensor
                fineFlow = self.layerOutputs['fineFlow']
                
                # Performs testing for mode 0
                # Saves each motion compensated image generated from 
                # from a frame set to result_dir
                # Note: Each testing image must have the same size
                for i in range(len(input_)):
                    result, courseF, fineF = self.sess.run([self.pred,  
                                                            courseFlow, fineFlow],
                                                           feed_dict={self.images_curr_prev: input_[i].reshape(1,
                                                                      self.h, self.w, 2*self.c_dim)})

                    # result = self.pred.eval({self.images_curr_prev:
                                             # input_[i].reshape(1,
                                             # self.h, self.w, 2*self.c_dim)})
    
                    original = input_[i].reshape(1, self.h, self.w, 2*self.c_dim)
                    original = original[0, :, :, 0:self.c_dim]
                    
                    errorMap = 1 - (original - result)
                    
                    # Computes mean error and appends to errorLst for 
                    # mean error computation
                    error = np.mean(np.square(original-result), axis=None)
                    errorLst.append(error)
                    print('Error on frame set ', i, ' : ', error)
                    
                    x = np.squeeze(result)
                    errorMapOut = np.squeeze(errorMap)
                    
                    courseFOut = np.squeeze(courseF)
                    fineFOut = np.squeeze(fineF)
                    
                    # Computes norm of course flow
                    courseNorm = np.sqrt(np.square(courseFOut[:, :, 0]) +
                                 np.square(courseFOut[:, :, 1]))
                    
                    # Computes norm of fine flow
                    fineNorm = np.sqrt(np.square(fineFOut[:, :, 0]) +
                                 np.square(fineFOut[:, :, 1]))

                    # Inverts norm maps for better view
                    courseNorm = 1 - courseNorm
                    fineNorm = 1 - fineNorm
                    
                    # Inverts course and fine flow maps
                    courseFOut = 1 - courseFOut
                    fineFOut = 1 - fineFOut
                                        
                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir+'/result'+str(i) + '.png', config)
                    imsave(errorMapOut, config.result_dir + '/result_errorMap' + str(i) + '.png', config)
                    imsave(courseFOut[:, :, 0], config.result_dir + '/result_courseMap0_' + str(i) + '.png', config)
                    imsave(courseFOut[:, :, 1], config.result_dir + '/result_courseMap1_' + str(i) + '.png', config)
                    imsave(fineFOut[:, :, 0], config.result_dir + '/result_fineMap0_' + str(i) + '.png', config)
                    imsave(fineFOut[:, :, 1], config.result_dir + '/result_fineMap1_' + str(i) + '.png', config)
                    imsave(courseNorm, config.result_dir + '/result_courseNorm_' + str(i) + '.png', config)
                    imsave(fineNorm, config.result_dir + '/result_fineNorm_' + str(i) + '.png', config)
                print('Mean Avg Error: ', np.mean(errorLst))
            elif self.train_mode == 1:
                
                # Performs testing for mode 1
                # Saves hr images for each input image to resut_dir
                for i in range(len(input_)):
                    result = self.pred.eval({self.images_in: input_[i].reshape(1, self.h, self.w, self.c_dim)})
                    
                    # Obtains original input image from input_
                    # orgInput = input_[i].reshape(self.h,
                                        # self.w, self.c_dim)
                    
                    # Denormalizes input image
                    # orgInput = orgInput * 255.0
                    
                    # Removes batch size axis from tensor
                    x = np.squeeze(result)

                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir+'/result'+str(i)+'.png',
                           config)
                    
                    # Loads output image
                    # data_LR = glob.glob('./result/result'+str(i)+'.png')
                    # lr = cv2.imread(data_LR[0])
                    
                    # print('Super resolution shape: ', np.shape(x))
                    # print('Original input shape: ', np.shape(orgInput))
                    
                    # Computes low resolution image from super resolution image
                    # by downscaling by self.scale
                    #lr = cv2.resize(lr, None,fx = 1.0/self.scale, 
                                            # fy = 1.0/self.scale,
                                            # interpolation = cv2.INTER_CUBIC)
                    
                    # Computes and prints PSNR ratio 
                    # Appends psnr val to psnrLst for mean computation
                    # psnrVal = psnr(lr, orgInput, scale = self.scale)
                    # psnrLst.append(psnrVal)
                    # print('Image ', i, ' PSNR: ', psnrVal )
                    
                    # Prints shape of output image and saves output image
                    
                    
                #print('Average PSNR: ', np.mean(psnrLst))
            elif self.train_mode == 2:
                
                # Performs testing for mode 2
                # Saves each hr images generated from a consecutive frame set
                # to result_dir
                for i in range(len(input_)):
                    
                    # Prepares current and previous frames
                    curr_prev = input_[i, :, :, 0:2*self.c_dim].reshape(1, self.h, self.w, 2*self.c_dim)
                    
                    # Prepares stack of current and next frames
                    curr_next = np.concatenate( (input_[i, :, :, 0:self.c_dim],
                                                 input_[i, :, :,
                                                   2*self.c_dim:3*self.c_dim]),
                                                axis=2).reshape(1,
                                                self.h, self.w, 2*self.c_dim)
    
                    # Fetches normalized original image from input_
                    # and restores pixel values to between 0-255
                    ''' curr_img = input_[i, :, :, 0:self.c_dim].reshape(self.h,
                                         self.w, self.c_dim) '''
                    
                    # curr_img = curr_img * 255.0
                    
                    # Generates output image
                    result = self.pred.eval({self.images_prev_curr: curr_prev,
                                             self.images_next_curr: curr_next})

                    # Removes batch size dimension from result
                    # and denormalizes result image 
                    x = np.squeeze(result)
                    
                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir + '/result'+str(i) + '.png', config)
                    
                    # Loads output image
                    # data_LR = glob.glob('./result/result'+str(i)+'.png')
                    # lr = cv2.imread(data_LR[0])
                    
                    # Downscales output image for psnr computation
                    # lr = cv2.resize(lr, None,fx = 1.0/self.scale,
                                            # fy = 1.0/self.scale,
                                            # interpolation = cv2.INTER_CUBIC)
                    
                    # Computes psnr and appends psnr to psnrLst for mean
                    # computation
                    # psnrVal = psnr(lr, curr_img, scale = self.scale)
                    # psnrLst.append(psnrVal)
                    # print('Image ', i, ' PSNR: ', psnrVal )
                    
                    # print(lr)
                    # print(curr_img)
                # print('Average PSNR: ', np.mean(psnrLst))
            elif self.train_mode == 3:
                
                # Performs testing for mode 1
                # Saves hr images for each input image to resut_dir
                for i in range(len(input_)):
                    result = self.pred.eval({self.images_in: input_[i].reshape(1, self.h, self.w, self.c_dim)})
    
                    # Obtains original input image from input_
                    orgInput = input_[i].reshape(self.h, self.w, self.c_dim)
                    
                    # Denormalizes input image
                    orgInput = orgInput * 255.0
                    
                    x = np.squeeze(result)
                    
                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir+'/result'+str(i)+'.png',
                           config)
                    
                    # Loads output image
                    data_LR = glob.glob('./result/result'+str(i)+'.png')
                    lr = cv2.imread(data_LR[0])
                    
                    print('Super resolution shape: ', np.shape(x))
                    print('Original input shape: ', np.shape(orgInput))
                    
                    # Computes low resolution image from super resolution image
                    # by downscaling by self.scale
                    lr = cv2.resize(lr, None, fx=1.0/self.scale, fy=1.0/self.scale,
                                    interpolation=cv2.INTER_CUBIC)
                    
                    # Computes and prints PSNR ratio 
                    # Appends psnr val to psnrLst for mean computation
                    psnrVal = psnr(lr, orgInput, scale=self.scale)
                    psnrLst.append(psnrVal)
                    print('Image ', i, ' PSNR: ', psnrVal)
                    
                    # Prints shape of output image and saves output image
                print('Average PSNR: ', np.mean(psnrLst))
            elif self.train_mode == 4:
                
                for i in range(len(input_)):
                    
                    # Upscales image using bicubic interpolatiobn
                    inp = input_[i].reshape(self.h, self.w, self.c_dim)
                    inp = sp.misc.imresize(inp,
                                           (self.h*self.scale,
                                            self.w*self.scale), interp='bicubic')
                    
                    result = self.pred.eval({self.images_in: inp.reshape(1,
                                             self.h*self.scale, self.w*self.scale, self.c_dim)})

                    x = np.squeeze(result)
                    # Filters image values to be in [0,1]
                    # x = np.clip(np.squeeze(result), 0, 1)
                    
                    print('Saving image ' , i)
                    print('Shape of output image: ', x.shape)
                    
                    imsave(x, config.result_dir+'/result'+str(i)+'.png',
                           config)
            elif self.train_mode == 5:
                # Performs testing for mode 2
                # Saves each hr images generated from a consecutive frame set
                # to result_dir
                
                count = 0
                
                for i in range(len(input_)):
                    
                    print('Working on dataset ' + str(i) + ' ...')
                    
                    # Gets folder name
                    folderName = os.path.basename(os.path.split(paths_[count])[0])
                    
                    folder = os.path.join(config.result_dir, folderName)
                    os.makedirs(folder)
                    for j in range(len(input_[i])):

                        # Prepares current and previous frames
                        curr_prev = input_[i][j, :, :, 0:2*self.c_dim].reshape(1,  self.h, self.w, 2*self.c_dim)
                        
                        # Prepares stack of current and next frames
                        curr_next = np.concatenate((input_[i][j, :, :, 0:self.c_dim],
                                                    input_[i][j, :, :,
                                                              2*self.c_dim:3*self.c_dim]), axis=2).reshape(1,
                                                                                                           self.h,
                                                                                                           self.w,
                                                                                                           2*self.c_dim)

                        # Generates output image
                        result = self.pred.eval({self.images_prev_curr: curr_prev,
                                                 self.images_next_curr: curr_next})
    
                        # Removes batch size dimension from result
                        # and denormalizes result image 
                        x = np.squeeze(result)
                        
                        print('Shape of output image: ', x.shape)
                        imsave(x, folder + '//result'+str(j) + '.png', config)
                        count = count + 1
            elif self.train_mode == 6:        
                
                count = 0
                for i in range(len(input_)):
                    
                    print('Working on dataset ' + str(i) + ' ...')
                    
                    # Gets folder name
                    folderName = os.path.basename(os.path.split(paths_[count])[0])
                    
                    folder = os.path.join(config.result_dir, folderName)
                    os.makedirs(folder)
                    
                    for j in range(len(input_[i])):
                        result = self.pred.eval({self.images_in: input_[i][j].reshape(1, self.h, self.w, self.c_dim)})

                        # Removes batch size axis from tensor
                        x = np.squeeze(result)

                        print('Shape of output image: ', x.shape)
                        imsave(x, folder + '//result'+str(j)+'.png',
                               config)
                        count = count + 1
     
                
            
    '''
    This method performs model loading given checkpoint_dir. Model
    is loaded based on self.image_size and self.scale from
    different directories. Different models are loaded based on
    self.train_mode. See class __init__() method for a description
    of different training modes.
    
    Input:
        checkpoint_dir: directory to load checkpoint
    
    '''
    def load(self, checkpoint_d):
        """
            This method loads the checkpoint for training or testing modes.

            Inputs:
            checkpoint_d: path of checkpoint to be loaded
        """
        
        print("\nReading Checkpoints.....\n\n")

        model_dir = ""
        
        # gives model name training data size and scale based on training mode
        if self.train_mode == 0:

            # Change model_dir to "stMC" in the future when doing fresh training
            model_dir = "%s_%s_%s" % ("espcn", self.image_size, self.scale)
        elif self.train_mode == 1 or self.train_mode == 6:

            # Change model_dir to "espcn" in the future when doing fresh training
            model_dir = "%s_%s_%s" % ("vespcn_subpixel_no_mc",
                                      self.image_size, self.scale)
        elif self.train_mode == 2 or self.train_mode == 5:
            model_dir = "%s_%s_%s" % ("vespcn",
                                      self.image_size, self.scale)
        elif self.train_mode == 4:
            model_dir = "%s_%s_%s" % ("srcnn",
                                      self.image_size, self.scale)

        checkpoint_d = os.path.join(checkpoint_d, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_d)
        
        # Checks if checkpoints exists 
        if ckpt and ckpt.model_checkpoint_path:
            
            # converts unicode to string
            ckpt_path = str(ckpt.model_checkpoint_path)
            
            # Loads model from ckt_path
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n" % ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")
            if self.train_mode == 5 or self.train_mode == 6:
                exit()
    
    '''
    This method performs model saving given checkpoint_dir. Model
    is saved based on self.image_size and self.scale into different
    directories, Different models are saved based on self.train_mode.
    See class __init__() method for a description of different training modes. 
    
    Input:
        checkpoint_dir: directory to load checkpoint
    
    '''
    def save(self, checkpoint_d, step):

        """
            This method saves the checkpoint for training or testing modes.

            Inputs:
            checkpoint_d: path of checkpoint to be saved
            step: step in training

        """
        
        model_name = ""
        model_dir = ""
        
        # gives model name by training data size and scale
        if self.train_mode == 0:

            # Change this to SpatialTransformerMC.model in the future
            # when doing fresh training model_dir: "stMC"
            model_name = "ESPCN.model"
            model_dir = "%s_%s_%s" % ("espcn", self.image_size, self.scale)
        elif self.train_mode == 1:

            # Change this to ESPCN.model in the future
            # when doing fresh training model_dir "espcn"
            model_name = "VESPCN_Subpixel_NO_MC.model"
            model_dir = "%s_%s_%s" % ("vespcn_subpixel_no_mc",
                                      self.image_size, self.scale)
        elif self.train_mode == 2:
            model_name = "VESPCN.model"
            model_dir = "%s_%s_%s" % ("vespcn",
                                      self.image_size, self.scale)
        elif self.train_mode == 4:
            model_name = "SRCNN.model"
            model_dir = "%s_%s_%s" % ("srcnn",
                                      self.image_size, self.scale)
            
        checkpoint_d = os.path.join(checkpoint_d, model_dir)
        
        # Checks if model checkpoint directory exists
        if not os.path.exists(checkpoint_d):
            os.makedirs(checkpoint_d)
        
        # Saves model to checkpoint_dir
        self.saver.save(self.sess,
                        os.path.join(checkpoint_d, model_name),
                        global_step=step)
