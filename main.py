import tensorflow as tf
import numpy as np
import math
import time
import os
import glob
import cv2
import scipy as sp
from model import ESPCN
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

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epoch", 1200, "Number of epoch")
flags.DEFINE_integer("steps_per_epoch", 3, "Steps per epoch")
flags.DEFINE_integer("image_size", 32, "The size of image input")
flags.DEFINE_integer("c_dim", 3, "The size of channel")
flags.DEFINE_boolean("is_train", True, "if training")
flags.DEFINE_integer("train_mode", 0, "0: Spatial Transformer 1: VESPSCN No MC\
                     2: VESPCN 3: Bicubic (No Training Required) 4: SRCNN \
                     5: Multi-Dir mode for testing mode 2 6: Multi-Dir mode \
                     for testing mode 1")
flags.DEFINE_integer("scale", 3,
                     "the size of scale factor for pre-processing input image")
flags.DEFINE_integer("stride", 100, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint",
                    "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate")
flags.DEFINE_integer("batch_size", 8, "the size of batch")
flags.DEFINE_string("result_dir", "result", "Name of result directory")
flags.DEFINE_string("test_img", "", "test_img")
flags.DEFINE_boolean("load_existing_data", False,
                     "True iff existing hf data is loaded for training/testing")
flags.DEFINE_string("job_name", "", "ps/worker")
flags.DEFINE_integer("task_index", -1, "task index")
flags.DEFINE_string("ps_hosts", "", "ps-task hosts in cluster")
flags.DEFINE_string("worker_hosts", "", "worker-task hosts in cluster")


def prepare_data(config):

    # Prepares data if load_existing_data is False
    if not config.load_existing_data:
        input_setup(config)

    # Loads data from data_dir
    print('Loading data...')
    data_dir = checkpoint_dir(config)
    input_, label_, paths_ = read_data(data_dir, config)

    # Shuffles training data
    print('Shuffling data...')
    numData = np.arange(input_.shape[0])
    np.random.shuffle(numData)
    input_ = input_[numData]
    label_ = label_[numData]

    # Prepares frame sets for feeding into different spatial
    # transformers if training mode is 2
    if FLAGS.train_mode == 2:
        print("Preparing frames sets for spatial transformers...")

        curr_prev_imgs = input_[:, :, :, 0:(2 * config.c_dim)]
        curr_next_imgs = np.concatenate((input_[:, :, :,
                                                0:config.c_dim],
                                        input_[:, :, :,
                                        (2 * config.c_dim):
                                        (3 * config.c_dim)]),
                                        axis=3)

        curr_prev_imgs = tf.cast(curr_prev_imgs, tf.float32)
        curr_next_imgs = tf.cast(curr_next_imgs, tf.float32)
        label_ = tf.cast(label_, tf.float32)

        # Provides data in batch one at a time to tf.train.batch
        input_queue = tf.train.slice_input_producer([curr_prev_imgs, curr_next_imgs, label_], shuffle=False)
        x1, x2, y = tf.train.batch(input_queue, batch_size=FLAGS.batch_size)
        return x1, x2, y
    elif FLAGS.train_mode == 4:

        # Upscales input data using bicubic interpolation
        print('Upscaling training data using Bicubic Interpolation...')

        input_new = []
        for i in range(len(input_)):
            input_new.append(sp.misc.imresize(input_[i],
                                              (self.image_size * self.scale,
                                               self.image_size * self.scale), interp='bicubic'))
        input_ = np.array(input_new)

        input_ = tf.cast(input_, tf.float32)
        label_ = tf.cast(label_, tf.float32)

        # Provides data in batch one at a time to tf.train.batch
        input_queue = tf.train.slice_input_producer([input_, label_], shuffle=False)
        x1, y = tf.train.batch(input_queue, batch_size=FLAGS.batch_size)
        return x1, y

    else:
        input_ = tf.cast(input_, tf.float32)
        label_ = tf.cast(label_, tf.float32)

        # Provides data in batch one at a time to tf.train.batch
        input_queue = tf.train.slice_input_producer([input_, label_], shuffle=False)
        x1, y = tf.train.batch(input_queue, batch_size=FLAGS.batch_size)
        return x1, y


def run_train_epochs(target1, cfg):

    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.steps_per_epoch)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    # master="grpc://" + worker_hosts[FLAGS.task_index]
    # if_chief: 制定task_index为0的任务为主任务，用于负责变量初始化、做checkpoint、保存summary和复原
    # 定义计算服务器需要运行的操作。在所有的计算服务器中有一个是主计算服务器。
    # 它除了负责计算反向传播的结果，它还负责输出日志和保存模型
    with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.checkpoint_dir,
                                           hooks=hooks,
                                           master=target1,
                                           config=cfg,
                                           is_chief=(FLAGS.task_index == 0),
                                           ) as sess:
        while not sess.should_stop():
            espcn.train(FLAGS, sess)



if __name__ == '__main__':

    # Checks console arguments are valid
    if FLAGS.job_name == "" or FLAGS.job_name is None:
        print("Error: Job name must be specified as ps or worker.")
        exit(1)

    if FLAGS.task_index < 0 or FLAGS.task_index is None:
        print('Error: task_index is negative.')
        exit(1)

    if str(FLAGS.ps_hosts).replace(' ', '') == '' or FLAGS.ps_hosts is None:
        print('Error: ps_hosts must be specified.')
        exit(1)

    if str(FLAGS.worker_hosts).replace(' ', '') == '' or FLAGS.worker_hosts is None:
        print('Error: worker_hosts must be specified.')
        exit(1)

    # Creates a cluster from the parameter server and worker hosts.
    cluster_spec = tf.train.ClusterSpec({
        "ps": FLAGS.ps_hosts.replace(' ', '').split(","),
        "worker": FLAGS.worker_hosts.replace(' ', '').split(",")
    })

    # Creates and starts a server for the local task.
    server = tf.train.Server(
        cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index, start=True)

    # # 参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程。
    # server.join()会一直停在这条语句上
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
        print('Worker Device: ', worker_device)

        device, target = (tf.train.replica_device_setter(
                        worker_device=worker_device,
                        cluster=cluster_spec),
                        server.target)

        #config = tf.ConfigProto(log_device_placement=True)
        #config.gpu_options.allow_growth = True

        # Checks if train mode is 3 and training is on
        if FLAGS.train_mode == 3 and FLAGS.is_train:
            print('Error: Bicubic Mode does not require training')
            exit(1)
        elif FLAGS.train_mode == 5 and FLAGS.is_train:
            print('Error: Multi-Dir testing mode for Mode 2 does not require training')
            exit(1)
        elif FLAGS.train_mode == 6 and FLAGS.is_train:
            print('Error: Multi-Dir testing mode for Mode 1 does not require training')
            exit(1)

        with tf.device(device):

            # Prepares data based on is_train and train_mode

            DataList = []

            if FLAGS.train_mode == 2:
                xx1, xx2, yy = prepare_data(FLAGS)
                DataList = [xx1, xx2, yy]
            else:
                xx1, yy = prepare_data(FLAGS)
                DataList = [xx1, yy]

            espcn = ESPCN(
                image_size=FLAGS.image_size,
                is_train=FLAGS.is_train,
                train_mode=FLAGS.train_mode,
                scale=FLAGS.scale,
                c_dim=FLAGS.c_dim,
                batch_size=FLAGS.batch_size,
                load_existing_data=FLAGS.load_existing_data,
                device=device,
                learn_rate=FLAGS.learning_rate,
                data_list=DataList)

            # 通过设置log_device_placement选项来记录operations 和 Tensor 被指派到哪个设备上运行
            config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
            )

            run_train_epochs(target, config)
