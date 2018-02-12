import tensorflow as tf
from model import ESPCN
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epoch", 400, "Number of epoch")
flags.DEFINE_integer("image_size", 32, "The size of image input")
flags.DEFINE_integer("c_dim", 3, "The size of channel")
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_integer("train_mode", 0, "0 spatial transformer 1 subpixel net 2 training in unison")
flags.DEFINE_integer("scale", 3, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 32, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-4 , "The learning rate")
flags.DEFINE_integer("batch_size", 128, "the size of batch")
flags.DEFINE_string("result_dir", "result", "Name of result directory")
flags.DEFINE_string("test_img", "", "test_img")


def main(_): #?
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        espcn = ESPCN(sess,
                      image_size = FLAGS.image_size,
                      is_train = FLAGS.is_train,
                      train_mode = FLAGS.train_mode,
                      scale = FLAGS.scale,
                      c_dim = FLAGS.c_dim,
                      batch_size = FLAGS.batch_size,
                      test_img = FLAGS.test_img,
                      config=config
                      )
        
        
        espcn.train(FLAGS)
        
if __name__=='__main__':
    
    # parses command argument then calls the main function
    tf.app.run() 
