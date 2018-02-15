import tensorflow as tf
import numpy as np
import math
import time
import os

from utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    checkimage,
    imsave,
    imread,
    load_data,
    preprocess,
)
from PSNR import psnr

class ESPCN(object):
    
    '''
       This method is contructor for class ESPSCN.
       Given: 
            sess: Session object for this model
            image_size: size of image for training
            is_train: True iff training
            train_mode: 0 is spatial transformer only
                        1 is single frame ESPSCN
                        2 is training in unison using VESPCN
            scale: upscaling ratio for super resolution
            batch_size: batch size for training
            c_dim: number of channels of each input image
            config: config object
       Returns: None
    '''                   
    def __init__(self,
                 sess,
                 image_size,
                 is_train,
                 train_mode,
                 scale,
                 batch_size,
                 c_dim,
                 config
                 ):
        
        self.sess = sess
        self.image_size = image_size
        self.is_train = is_train
        self.c_dim = c_dim
        self.scale = scale
        self.train_mode = train_mode
        self.batch_size = batch_size
        self.config = config
        self.build_model()
    
    '''
       This method builds network based on training mode. Placeholders are
       setup based on training mode and is_train. Predictions are obtained
       by calling method model. Saver is initialized and loss functions are
       initialized based on training mode.
       
       Given: None
       Returns: None
    '''
    def build_model(self):
        
        
        if self.is_train:
            
            # Prepares placeholder based on train_mode used
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
            elif self.train_mode == 1:
                
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
            
            # Computes shape of placeholder for image feed from sample image
            # loaded
            print('Train Mode:' , self.train_mode)
            data = load_data(self.is_train, self.train_mode)
            input_ = imread(data[0][0])       
            self.h, self.w, c = input_.shape
            
            # Prepares placeholders based on training_mode used
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
            elif self.train_mode == 1:
                
                # Sets up placeholders for training subpixel net 
                # if train mode is 1
                self.images_in = tf.placeholder(tf.float32,
                                                [None,
                                                 self.h,
                                                 self.w,
                                                 self.c_dim],
                                                 name='images_in')
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
        
        if self.train_mode == 0 or self.train_mode == 1:
            self.pred = self.model()
        else:
            self.pred, self.imgPrev, self.imgNext = self.model()
        
        # Prepares loss function based on training mode
        if self.train_mode == 0:
            
            # Defines loss function for training a single spatial transformer
            self.loss = tf.reduce_sum(tf.square(self.labels - self.pred))
        elif self.train_mode == 1:
            
            # Defines loss function for training subpixel convnet
            self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        else:
            
            # Defines loss function for training in unison
            self.loss = tf.reduce_mean(tf.square(self.labels - self.pred)) \
            + tf.reduce_mean(tf.square(self.imgPrev-self.images_prev_curr[:, :, :, 0:self.c_dim])) \
            + tf.reduce_mean(tf.square(self.imgNext-self.images_prev_curr[:, :, :, 0:self.c_dim])) 

        self.saver = tf.train.Saver() # To save checkpoint
        
    '''
    This method constructs spatial transformer given frameSet.
    
    Inputs:
    frameSet: frameSet containing 2 images: target frame and neighbouring frame
              Tensor of dimension: [nBatch, imgH, imgW, 6]
              
    reuse: True iff Weights are the same as network where reuse is false
    
    Returns: Output image of this spatial transformer network
             Tensor of dimension: [nBatch, imgH, imgW, 3]
    '''
    def spatial_transformer(self, frameSet, reuse=False):
        
        # Zero initialization
        biasInitializer = tf.zeros_initializer()
        
        # Orthogonal initialization with gain sqrt(2)
        weight_init = tf.orthogonal_initializer(np.sqrt(2))
        
        # Course flow
        t1_course_l1 = tf.layers.conv2d(frameSet,  24, 5, padding = 'same',
                                        strides = (2,2),
                                        activation=tf.nn.relu,
                                        kernel_initializer = weight_init, 
                                        bias_initializer = biasInitializer,
                                        name = 't1_course_l1', reuse=reuse) 

        t1_course_l2 = tf.layers.conv2d(t1_course_l1,  24, 5, padding='same',
                                        strides = (1,1),
                                        activation=tf.nn.relu,
                                        kernel_initializer = weight_init,
                                        bias_initializer = biasInitializer,
                                        name = 't1_course_l2', reuse=reuse)
        t1_course_l3 = tf.layers.conv2d(t1_course_l2,  24, 5, padding='same',
                                        strides = (2,2),
                                        activation=tf.nn.relu,
                                        kernel_initializer = weight_init,
                                        bias_initializer = biasInitializer,
                                        name = 't1_course_l3', reuse=reuse)
        t1_course_l4 = tf.layers.conv2d(t1_course_l3,  24, 3, padding='same',
                                        strides = (1,1),
                                        activation=tf.nn.relu,
                                        kernel_initializer = weight_init,
                                        bias_initializer = biasInitializer,
                                        name = 't1_course_l4', reuse=reuse)
        t1_course_l5 = tf.layers.conv2d(t1_course_l4,  32, 3, padding='same',
                                        strides = (1,1),
                                        activation=tf.nn.tanh,
                                        kernel_initializer = weight_init,
                                        bias_initializer = biasInitializer,
                                        name = 't1_course_l5', reuse=reuse)
        
        # Defines Course Flow Output
        # Output shape: (-1, l, w, 2)
        t1_course_out = self.PS2(t1_course_l5, 4, 2)
        
        # Course Warping
        # Gets target image to be warped
        targetImg = frameSet[:, :, :, self.c_dim:self.c_dim*2]
        
        # Generates tensor of dimension [-1, h, w, 3+2]
        t1_course_warp_in = tf.concat([targetImg, t1_course_out], 3)
        
        # Applies warping using 2D convolution layer to estimate image at time
        # t=t
        # Kernel size 3 is used to apply flow to image based on neighbouring
        # flows of pixel
        t1_course_warp = tf.layers.conv2d(t1_course_warp_in,  3, 3,
                                          padding='same',
                                          activation=tf.nn.tanh,
                                          kernel_initializer = weight_init,
                                          bias_initializer = biasInitializer,
                                          name = 't1_course_warp', reuse=reuse)
        
        # Fine flow 
        # Combines images input, course flow estimation, 
        # and course image estimation along dimension 3
        t1_fine_in = tf.concat([frameSet, t1_course_warp,
                                t1_course_out], 3)
        
        t1_fine_l1 = tf.layers.conv2d(t1_fine_in,  24, 5, padding='same',
                                      strides = (2,2),
                                      activation=tf.nn.relu,
                                      kernel_initializer = weight_init,
                                      bias_initializer = biasInitializer,
                                      name = 't1_fine_l1', reuse=reuse) 
        
        t1_fine_l2 = tf.layers.conv2d(t1_fine_l1,  24, 3, padding='same',
                                      strides = (1,1),
                                      activation=tf.nn.relu,
                                      kernel_initializer = weight_init,
                                      bias_initializer = biasInitializer,
                                      name = 't1_fine_l2', reuse=reuse)
        
        t1_fine_l3 = tf.layers.conv2d(t1_fine_l2,  24, 3, padding='same',
                                      strides = (1,1),
                                      activation=tf.nn.relu,
                                      kernel_initializer = weight_init,
                                      bias_initializer = biasInitializer,
                                      name = 't1_fine_l3', reuse=reuse)
        
        t1_fine_l4 = tf.layers.conv2d(t1_fine_l3,  24, 3, padding='same',
                                      strides = (1,1),
                                      activation=tf.nn.relu,
                                      kernel_initializer = weight_init,
                                      bias_initializer = biasInitializer,
                                      name = 't1_fine_l4', reuse=reuse)
        
        t1_fine_l5 = tf.layers.conv2d(t1_fine_l4,  8, 3, padding='same',
                                      strides = (1,1),
                                      activation=tf.nn.tanh,
                                      kernel_initializer = weight_init,
                                      bias_initializer = biasInitializer,
                                      name = 't1_fine_l5', reuse=reuse)
        
        # Output shape(-1, l, w, 2)
        t1_fine_out = self.PS2(t1_fine_l5, 2, 2)
        
        # Combines fine flow and course flow estimates
        # Output shape(-1, l, w, 2)
        t1_combined_flow = t1_course_out + t1_fine_out
        
        # Fine Warping
        # Concatnates target image and combined flow channels
        t1_fine_warp_in = tf.concat([targetImg, t1_combined_flow], 3)
        
        # Applies warping using 2D convolution layer to estimate image at time
        # t=t
        # Kernel size 3 is used to apply flow based on neighbouring flows of
        # pixel
        # Output shape: (batchSize, h, w, c_dim)
        t1_fine_warp = tf.layers.conv2d(t1_fine_warp_in, 3, 3, padding='same',
                                        activation=tf.nn.tanh, 
                                        kernel_initializer = weight_init,
                                        bias_initializer = biasInitializer,
                                        name = 't1_fine_warp', reuse=reuse)
        
        # Resizes using billinear interpolation
        # Output shape: (batchSize, h, w, c_dim)
        return(t1_fine_warp)
        
    '''
    This method generates a network model given self.train_mode
    train mode = 0: model for 1 spatial transformer
    train mode = 1: model for single frame ESPSCN 
    train mode = 2: model for VESPCN with 2 spatial transformers taking 2 images
                    each
    
    Returns: Output of network if train mode is 0 or 1
             Output of network and output of 2 spatial transformers of network
             if train mode is 2
    '''
    def model(self):
        
        
       # Generates motion compensated images from previous and next images
       # using 2 spatial transformers
           
       # Initializes spatial transformer if training mode is 0 or 2
       if self.train_mode == 2:
           imgPrev = self.spatial_transformer(self.images_prev_curr,
                                              reuse = False)
           imgNext = self.spatial_transformer(self.images_next_curr,
                                              reuse = True)
               
           targetImg = self.images_prev_curr[:, :, :, 0:self.c_dim]
           imgSet = tf.concat([imgPrev, targetImg, imgNext], 3)
       elif self.train_mode == 0:
           motionCompensatedImgOut = self.spatial_transformer(self.
                                                              images_curr_prev,
                                                              reuse = False)
       else:
           imgSet = self.images_in
       
        
       wInitializer1 = tf.random_normal_initializer(stddev=np.sqrt(2.0/25/3))
       wInitializer2 = tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64))
       wInitializer3 = tf.random_normal_initializer(stddev=np.sqrt(2.0/9/32))
       biasInitializer = tf.zeros_initializer()
       
       
       if self.train_mode == 2:
           
           # Connects early fusion network with spatial transformer 
           # and subpixel convnet
           EarlyFusion =  tf.layers.conv2d(imgSet,  3, 3, padding='same',
                                           activation=tf.nn.relu,
                                           kernel_initializer = wInitializer1,
                                           bias_initializer = biasInitializer,
                                           name = 'EF1')
           subPixelIn = EarlyFusion
       elif self.train_mode == 1:
           
           # Connects subpixel convnet to placeholder for feeding single images
           subPixelIn = imgSet
    
       # TO DO: Enable all layers in every step but 
       # adjust inputs and outputs to layers depending on
       # mode so the entire model is accessible through checkpoint
       # Builds subpixel net if train mode is 1 or 2
       if self.train_mode == 1 or self.train_mode == 2:
          
           conv1 = tf.layers.conv2d(subPixelIn,  64, 3, padding='same',
                                    activation=tf.nn.relu,
                                    kernel_initializer = wInitializer1,
                                    bias_initializer = biasInitializer,
                                    name = 'subPixelL1')
           conv2 = tf.layers.conv2d(conv1,  32, 3, padding='same',
                                    activation=tf.nn.relu,
                                    kernel_initializer = wInitializer2,
                                    bias_initializer = biasInitializer,
                                    name = 'subPixelL2')
           conv3 = tf.layers.conv2d(conv2,
                                    self.c_dim * self.scale * self.scale,
                                    3, padding='same', activation=None,
                                    kernel_initializer = wInitializer3,
                                    bias_initializer = biasInitializer,
                                    name = 'subPixelL3')

           ps = self.PS(conv3, self.scale)
       
       # Returns network output given self.train_mode
       if self.train_mode == 0:
           return motionCompensatedImgOut
       elif self.train_mode == 1:
           return tf.nn.tanh(ps)
       else:
           return (tf.nn.tanh(ps), imgPrev, imgNext) 
    
    '''
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
    '''
    #NOTE: train with with batch size
    def _phase_shift(self, I, r):
        
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        
        # a, [bsize, b, r, r]
        X = tf.split(X, a, 1)  
        
        # bsize, b, a*r, r
        X = tf.concat([tf.squeeze(x) for x in X], 2)  
        
        # b, [bsize, a*r, r]
        X = tf.split(X, b, 1)  
        
        # bsize, a*r, b*r
        X = tf.concat([tf.squeeze(x) for x in X], 2)  
        return tf.reshape(X, (self.batch_size, a*r, b*r, 1))

    '''
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
    '''
    # NOTE:test without batchsize
    def _phase_shift_test(self, I, r):
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
        
    '''
       Performs phase shift operation for tensor of dimension
       (batch_size, img_height, img_width, 3*r*r)
       
       Inputs:
       X: tensor of dimension (batch_size, img_height, img_width, 3*r*r)
       r: upscaling factor
       c_dim: c_dim of X
       
       Returns:
       Tensor of shape (batchs_size, img_height*r, img_width*r, 3)
    '''
    def PS(self, X, r):
        
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            
            # Does concat RGB
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) 
        else:
            
            # Does concat RGB
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) 
        return X
    
    '''
       Performs phase shift operation for tensor of dimension
       (batch_size, img_height, img_width, c_dim)
       
       Inputs:
       X: tensor of dimension (batch_size, img_height, img_width, c_dim*r*r)
       r: upscaling factor
       c_dim: c_dim of X
       
       Returns:
       Tensor of shape (batchs_size, img_height*r, img_width*r, c_dim)
    '''
    def PS2(self, X, r, c_dim):
        
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
    
    '''
       This method performs training / testing operations given config.
       Training / Testing is supported for all 3 modes from self.train_mode.
       
       See class __init__() for additional a description of training modes.
       
       Input:
            config: config object of this class
       
    '''
    def train(self, config):
        
        # NOTE : if train, the nx, ny are ingnored
        input_setup(config)

        data_dir = checkpoint_dir(config)
        
        input_, label_ = read_data(data_dir)
        
        print('Input and Label Shapes:')
        print(input_.shape, label_.shape)

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=
                                     config.learning_rate).minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())

        counter = 0
        time_ = time.time()

        self.load(config.checkpoint_dir)
        # Train
        if config.is_train:
            print("Now Start Training...")
            for ep in range(config.epoch):
                
                # Runs by batch images
                batch_idxs = len(input_) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = input_[idx * config.batch_size 
                                          : (idx + 1) * config.batch_size]
                    batch_labels = label_[idx * config.batch_size 
                                          : (idx + 1) * config.batch_size]
                    counter += 1
                    
                    # Feeds data into placeholders based on self.training_mode
                    if(config.train_mode == 0):
                        _, err = self.sess.run([self.train_op, self.loss],
                                               feed_dict = \
                                               {self.images_curr_prev: batch_images,
                                                self.labels: batch_labels})
                    elif(config.train_mode == 1):
                        _, err = self.sess.run([self.train_op, self.loss],
                                               feed_dict={self.images_in: batch_images,
                                                          self.labels: batch_labels})
                    elif(config.train_mode == 2):
                        curr_prev = batch_images[:, :, :, 0:2*self.c_dim]
                        curr_next = np.concatenate( (batch_images[:, :, :, 0:self.c_dim ],
                                                batch_images[:, :, :,
                                                             2*self.c_dim:3*self.c_dim]),
                                                              axis = 3)
                        _, err = self.sess.run([self.train_op, self.loss],
                                               feed_dict={self.images_prev_curr: curr_prev,
                                                           self.images_next_curr: curr_next,
                                                          self.labels: batch_labels})

                    if counter % 10 == 0:
                        print("Epoch: ", (ep+1), " Step: ", counter, " Time: ",
                              (time.time()-time_), " Loss: ", err)
                        
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
        # Test
        else:
            print("Now Start Testing...")
            
            if (self.train_mode==0):
                
                # Performs testing for mode 0
                # Saves each motion compensated image generated from 
                # from a frame set to result_dir
                # Note: Each testing image must have the same size
                for i in range(len(input_)):
                    result = self.pred.eval({self.images_curr_prev: input_[i].reshape(1,
                                             self.h, self.w, 2*self.c_dim)})
                    original = input_[i].reshape(1, self.h, self.w, 2*self.c_dim)
                    original = original[0, :, :, 0:self.c_dim]
                    
                    print('Error on frame set ', i, ' : ',
                          np.sum(np.square(original-result), axis=None))
                    x = np.squeeze(result)
                    
                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir+'/result'+str(i)+'.png', config)
                
            elif self.train_mode == 1:
                
                # Performs testing for mode 1
                # Saves hr images for each input image to resut_dir
                for i in range(len(input_)):
                    result = self.pred.eval({self.images_in: input_[i].reshape(1,
                                             self.h, self.w, self.c_dim)})
    
                    x = np.squeeze(result)
                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir+'/result'+str(i)+'.png',
                           config)
            elif self.train_mode == 2:
                
                # Performs testing for mode 2
                # Saves each hr images generated from a consecutive frame set
                # to result_dir
                for i in range(len(input_)):
                    
                    # Prepares current and previous frames
                    curr_prev = input_[i, :, :, 0:2*self.c_dim].reshape(1,
                                             self.h, self.w, 2*self.c_dim)
                    
                    # Prepares stack of current and next frames
                    curr_next = np.concatenate( (input_[i, :, :, 0:self.c_dim],
                                            input_[i,  :, :, 
                                                   2*self.c_dim:3*self.c_dim]),
                                                axis=2).reshape(1,
                                             self.h, self.w, 2*self.c_dim)
                            
                    
                    result = self.pred.eval({self.images_prev_curr: curr_prev,
                                             self.images_next_curr: curr_next})
    
                    x = np.squeeze(result)
                    print('Shape of output image: ', x.shape)
                    imsave(x, config.result_dir+'/result'+str(i)+'.png',
                           config)
            
    '''
    This method performs model loading given checkpoint_dir. Model
    is loaded based on self.image_size and self.scale from
    different directories. Different models are loaded based on
    self.train_mode. See class __init__() method for a description
    of different training modes.
    
    Input:
        checkpoint_dir: directory to load checkpoint
    
    '''
    def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        
        print("\nReading Checkpoints.....\n\n")
        
        
        model_dir = ""
        
        # gives model name training data size and scale based on training mode
        if(self.train_mode == 0):
            model_dir = "%s_%s_%s" % ("espcn", self.image_size,self.scale)
        elif(self.train_mode == 1):
            model_dir = "%s_%s_%s" % ("espcn_subpixel",
                                      self.image_size,self.scale)
        elif(self.train_mode == 2):
            model_dir = "%s_%s_%s" % ("vespcn",
                                      self.image_size,self.scale)
            
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Checks if checkpoints exists 
        if ckpt and ckpt.model_checkpoint_path:
            
            # converts unicode to string
            ckpt_path = str(ckpt.model_checkpoint_path)
            
            # Loads model from ckt_path
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")
    
    '''
    This method performs model saving given checkpoint_dir. Model
    is saved based on self.image_size and self.scale into different
    directories, Different models are saved based on self.train_mode.
    See class __init__() method for a description of different training modes. 
    
    Input:
        checkpoint_dir: directory to load checkpoint
    
    '''
    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        
        model_name = ""
        model_dir = ""
        
        # gives model name by training data size and scale
        if (self.train_mode == 0):
            model_name = "ESPCN.model"
            model_dir = "%s_%s_%s" % ("espcn", self.image_size,self.scale)
        elif (self.train_mode == 1):
            model_name = "ESPCN_Subpixel.model"
            model_dir = "%s_%s_%s" % ("espcn_subpixel",
                                      self.image_size,self.scale)
        elif (self.train_mode == 2):
            model_name = "VESPCN.model"
            model_dir = "%s_%s_%s" % ("vespcn",
                                      self.image_size,self.scale)
            
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        # Checks if model checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)
        
        # Saves model to checkpoint_dir
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
