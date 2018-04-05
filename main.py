import tensorflow as tf
from model import ESPCN
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epoch", 1200, "Number of epoch")
flags.DEFINE_integer("image_size", 32, "The size of image input")
flags.DEFINE_integer("c_dim", 3, "The size of channel")
flags.DEFINE_boolean("is_train", True, "if training")
flags.DEFINE_integer("train_mode", 0, "0: Spatial Transformer 1: VESPSCN No MC\
                     2: VESPCN 3: Bicubic (No Training Required) 4: SRCNN \
                     5: Multi-Dir mode for testing mode 2 6: Multi-Dir mode \
                     for testing mode 1")
flags.DEFINE_integer("scale", 3,
                     "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 100, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint",
                    "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-4 , "The learning rate")
flags.DEFINE_integer("batch_size", 128, "the size of batch")
flags.DEFINE_string("result_dir", "result", "Name of result directory")
flags.DEFINE_string("test_img", "", "test_img")
flags.DEFINE_boolean("load_existing_data", False,
                    "True iff existing hf data is loaded for training/testing")


def main(_): #?
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Checks if train mode is 3, 5 or 6 and training mode is on
    if (FLAGS.train_mode == 3 and FLAGS.is_train):
        print('Error: Bicubic Mode does not require training')
        return
    elif (FLAGS.train_mode == 5 and FLAGS.is_train):
        print('Error: Multi-Dir testing mode for Mode 2 does not require training')
        return
    elif(FLAGS.train_mode == 6 and FLAGS.is_train):
        print('Error: Multi-Dir testing mode for Mode 1 does not require training')
        return
    
    with tf.Session(config=config) as sess:
        espcn = ESPCN(sess,
                      image_size = FLAGS.image_size,
                      is_train = FLAGS.is_train,
                      train_mode = FLAGS.train_mode,
                      scale = FLAGS.scale,
                      c_dim = FLAGS.c_dim,
                      batch_size = FLAGS.batch_size,
                      load_existing_data = FLAGS.load_existing_data,
                      config=config
                      )
    
        
        espcn.train(FLAGS)
        
if __name__=='__main__':
    
    # Parses command-line argument then calls the main function
    tf.app.run() 
