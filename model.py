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
    
    
    def __init__(self,
                 sess,
                 image_size,
                 is_train,
                 scale,
                 batch_size,
                 c_dim,
                 test_img,
                 ):
        
        self.sess = sess
        self.image_size = image_size
        self.is_train = is_train
        self.c_dim = c_dim
        self.scale = scale
        self.batch_size = batch_size
        self.test_img = test_img
        self.build_model()

    def build_model(self):

        if self.is_train:
            self.images_prev_curr = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 2*self.c_dim], name='images_prev_curr')
            self.images_next_curr = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 2*self.c_dim], name='images_next_curr')
            self.labels = tf.placeholder(tf.float32, [None, self.image_size * self.scale , self.image_size * self.scale, self.c_dim], name='labels')
        else:
            '''
                Because the test need to put image to model,
                so here we don't need do preprocess, so we set input as the same with preprocess output
            '''
            data = load_data(self.is_train, self.test_img)
            input_ = imread(data[0])       
            self.h, self.w, c = input_.shape
            
            
            # Defines placeholder for sets of 2 images at time t and t-1
            # Note: None means we accept any number of sets of 2 images
            #self.imgA = tf.placeholder(tf.float32, [None, self.h, self.w, self.c_dim*2], name = 'image0')
            
            # Defines placeholder for sets of 2 images at time t and t+1
            #self.imgB = tf.placeholder(tf.float32, [None, self.h, self.w, self.c_dim*2], name = 'image1')
            
            self.images_prev_curr = tf.placeholder(tf.float32, [None, self.h, self.w, 2*self.c_dim], name='images_prev_curr')
            self.images_next_curr = tf.placeholder(tf.float32, [None, self.h, self.w, 2*self.c_dim], name='images_next_curr')
            self.labels = tf.placeholder(tf.float32, [None, self.h * self.scale, self.w * self.scale, self.c_dim], name='labels')
            
        '''self.weights = {
            'EarlyFusionAw':  tf.Variable(tf.random_normal([1, 1, 2*self.c_dim, 3], stddev=np.sqrt(2.0/1/6)), name='EarlyFusionA')
            'EarlyFusionBw': tf.Variable(tf.random_normal([1, 1, 2*self.c_dim, 3], stddev=np.sqrt(2.0/1/6)), name='EarlyFusionB')
            
            'w1': tf.Variable(tf.random_normal([5, 5, self.c_dim, 64], stddev=np.sqrt(2.0/25/3)), name='w1'),
            'w2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=np.sqrt(2.0/9/64)), name='w2'),
            'w3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale ], stddev=np.sqrt(2.0/9/32)), name='w3')
        }
        
        self.biases = {
            'EarlyFusionAb': tf.Variable(tf.zeros([3], name='EarlyFusionAb'))
            'EarlyFusionBb': tf.Variable(tf.zeros([3], name='EarlyFusionBb'))
                
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale ], name='b3'))
        }'''
        
        
        self.pred = self.model()
        
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver() # To save checkpoint
    
    def spatial_transformer(self, frameSet):
        
        # Zero initialization
        biasInitializer = tf.zeros_initializer()
        
        # Orthogonal initialization with gain sqrt(2)
        weight_init = tf.orthogonal_initializer(np.sqrt(2))
        
        # Course flow
        t1_course_l1 = tf.layers.conv2d(frameSet,  24, 5, strides=(2, 2), padding='same', activation=tf.nn.relu, kernel_initializer = weight_init,  biasInitializer = biasInitializer) 
        t1_course_l2 = tf.layers.conv2d(t1_course_l1,  24, 5, padding='same', activation=tf.nn.relu, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        t1_course_l3 = tf.layers.conv2d(t1_course_l2,  24, 5, strides=(2, 2), padding='same', activation=tf.nn.relu, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        t1_course_l4 = tf.layers.conv2d(t1_course_l3,  24, 3, padding='same', activation=tf.nn.relu, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        t1_course_l5 = tf.layers.conv2d(t1_course_l4,  32, 3, padding='same', activation=tf.nn.tanh, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        
        # Output shape: (-1, l, w, 2)
        t1_course_out = tf.layers.conv2d(t1_course_l5, 2, 1, strides=(1, 1), padding='same', activation=tf.nn.tanh, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        
        # Course Warping
        
        # Gets target image to be warped
        targetImg = self.images.next_curr[:, :, :, 0:self.c_dim]
        
        # Generates tensor of dimension [-1, h, w, 3+2]
        t1_course_warp_in = tf.concat([targetImg, t1_course_out], 3)
        
        # Applies warping using 2D convolution layer to estimate image at time t=t
        # Kernel size 3 is used to estimate dI/dx and dI/dy from neighbouring pixel values
        t1_course_warp = tf.layers.conv2d(t1_course_warp_in,  3, 3, padding='same', activation=tf.nn.tanh, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        
        # Resizes using billinear interpolation
        # Output shape: (batchSize, h, w, c_dim)
        if self.is_train:
            t1_course_image_out = tf.image.resize_images(t1_course_warp, (self.image_size, self.image_size) )
        else:
            t1_course_image_out = tf.image.resize_images(t1_course_warp, (self.h, self.w) )
        
        # Fine flow
        
        # Combines images input, course flow estimation, and course image estimation along dimension 3
        t1_fine_in = tf.concat([frameSet, t1_course_out, t1_course_image_out], 3)
        
        t1_fine_l1 = tf.layers.conv2d(t1_fine_in,  24, 5, strides=(2, 2), padding='same', activation=tf.nn.relu, kernel_initializer = weight_init,  biasInitializer = biasInitializer) 
        t1_fine_l2 = tf.layers.conv2d(t1_fine_l1,  24, 3, padding='same', activation=tf.nn.relu, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        t1_fine_l3 = tf.layers.conv2d(t1_fine_l2,  24, 3, padding='same', activation=tf.nn.relu, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        t1_fine_l4 = tf.layers.conv2d(t1_fine_l3,  24, 3, padding='same', activation=tf.nn.relu, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        t1_fine_l5 = tf.layers.conv2d(t1_fine_l4,  8, 3, padding='same', activation=tf.nn.tanh, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        
        # Output shape(-1, l, w, 2)
        t1_fine_out = tf.layers.conv2d(t1_fine_l5, 2, 1, padding='same', activation=tf.nn.tanh, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        
        # Combines fine flow and course flow estimates
        # Output shape(-1, l, w, 2)
        t1_combined_flow = t1_course_out + t1_fine_out
        
        # Fine Warping
        
        # Concatnates target image and combined flow channels
        t1_fine_warp_in = tf.concat([targetImg, t1_combined_flow], 3)
        
        # Applies warping using 2D convolution layer to estimate image at time t=t
        # Kernel size 3 is used to estimate dI/dx and dI/dy from neighbouring pixel values
        # Output shape: (batchSize, h, w, c_dim)
        t1_fine_warp = tf.layers.conv2d(t1_fine_warp_in, 3, 3, padding='same', activation=tf.nn.tanh, kernel_initializer = weight_init,  biasInitializer = biasInitializer)
        
        # Resizes using billinear interpolation
        # Output shape: (batchSize, h, w, c_dim)
        if self.is_train:
            t1_image_out = tf.image.resize_images(t1_fine_warp, (self.image_size, self.image_size))
        else:
            t1_image_out = tf.image.resize_images(t1_fine_warp, (self.h, self.w))
            
        return(t1_image_out)
        
    def model(self):
        
        # Produces early fusion output from images at time t and t-1
        #fusionA = tf.nn.relu(tf.nn.conv2d(self.imgA, self.weights['EarlyFusionAw'], strides=[1,1,1,1], padding='SAME') + self.biases['EarlyFusionAb'])
        
        # Produces early fusion output from images at time t and t+1
        #fusionB = tf.nn.relu(tf.nn.conv2d(self.imgB, self.weights['EarlyFusionBw'], strides=[1,1,1,1], padding='SAME') + self.biases['EarlyFusionBb'])
        
        # Reshapes prev and next with current frame input sets 
        
       # Generates motion compensated images from previous and next images
       # using 2 spatial transformers
       imgPrev = self.spatial_transformer(self.images_prev_curr)
       imgNext = self.spatial_transformer(self.images_next_curr)
       
       targetImg = self.images.next_curr[:, :, :, 0:self.c_dim]
       
       # Concatenates motion compensated neighbouring frames and target frame
       imgSet = tf.concat([imgPrev, targetImg, imgNext], 3)
       
       wInitializer1 = tf.random_normal_initializer(stddev=np.sqrt(2.0/25/3))
       wInitializer2 = tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64))
       wInitializer3 = tf.random_normal_initializer(stddev=np.sqrt(2.0/9/32))
       biasInitializer = tf.zeros_initializer()
       
       EarlyFusion =  tf.layers.conv2d(imgSet,  3, 3, padding='same', activation=tf.nn.relu, kernel_initializer = wInitializer1,  biasInitializer = biasInitializer)
       
       conv1 = tf.layers.conv2d(EarlyFusion,  64, 5, padding='same', activation=tf.nn.relu, kernel_initializer = wInitializer1,  biasInitializer = biasInitializer)
       conv2 = tf.layers.conv2d(conv1,  32, 3, padding='same', activation=tf.nn.relu, kernel_initializer = wInitializer2,  biasInitializer = biasInitializer)
       conv3 = tf.layers.conv2d(conv2,  self.c_dim * self.scale * self.scale, 3, padding='same', activation=None, kernel_initializer = wInitializer3,  biasInitializer = biasInitializer)
       #conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
       #conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
       #conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3'] # This layer don't need ReLU
       ps = self.PS(conv3, self.scale)
       return tf.nn.tanh(ps)

    #NOTE: train with batch size 
    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (self.batch_size, a*r, b*r, 1))

    # NOTE:test without batchsize
    def _phase_shift_test(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a*r, b*r, 1))
        

    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) # Do the concat RGB
        else:
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) # Do the concat RGB
        return X
    
    '''
       Performs phase shift operation for tensor of dimension
       (batch_size, img_height, img_width, c_dim)
       
       Inputs:
       X: tensor of dimension (batch_size, img_height, img_width, c_dim)
       r: upscaling factor
       c_dim: c_dim of X
    '''
    def PS2(self, X, r, c_dim):
        # Main OP that you can arbitrarily use in you tensorflow code
        
        # Evenly splits Xc into 2 parts along axis 3 (# of channels)
        Xc = tf.split(X, c_dim, 3)
        if self.is_train:
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) # Do the concat RGB
        else:
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) # Do the concat RGB
        return X

    def train(self, config):
        
        # NOTE : if train, the nx, ny are ingnored
        input_setup(config)

        data_dir = checkpoint_dir(config)
        
        input_, label_ = read_data(data_dir)

        print(input_.shape, label_.shape)

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

        counter = 0
        time_ = time.time()

        self.load(config.checkpoint_dir)
        # Train
        if config.is_train:
            print("Now Start Training...")
            for ep in range(config.epoch):
                # Run by batch images
                batch_idxs = len(input_) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = input_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    batch_labels = label_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

                    if counter % 10 == 0:
                        print("Epoch: ", (ep+1), " Step: ", counter, " Time: ", (time.time()-time_), " Loss: ", err)
                        #print(label_[1] - self.pred.eval({self.images: input_})[1],'loss:]',err)
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
        # Test
        else:
            print("Now Start Testing...")
            result = self.pred.eval({self.images: input_[0].reshape(1, self.h, self.w, self.c_dim)})
            x = np.squeeze(result)
            checkimage(x)
            print(x.shape)
            imsave(x, config.result_dir+'/result.png', config)
    def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s_%s_%s" % ("espcn", self.image_size,self.scale)# give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Check the checkpoint is exist 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path) # convert the unicode to string
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")
    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        model_name = "ESPCN.model"
        model_dir = "%s_%s_%s" % ("espcn", self.image_size,self.scale)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
