# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf

import image_processing
import scipy.misc

# import model
import model_2 as model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps', 70000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('set', 'train',
                           """Either 'train' or 'validation'.""")

tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_float('learning_rate', 0.0000001,
                            """How many GPUs to use.""")

tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_boolean('is_training', True,
                            """If set, dropout """)
                           
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '/data/huozengwei/train_dir/biwi_4/',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/data/huozengwei/train_dir/aflw_2016/aflw_1/',
                           """Directory where to read model checkpoints.""")


def tower_loss(scope, images, labels):
    train_output = model.inference(images,FLAGS.is_training)
    _ = model.losses(train_output, labels)   

    losses = tf.get_collection('losses', scope)

    total_loss = tf.add_n(losses,name='total_loss')

    return total_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g,_ in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0,values=grads)
        grad = tf.reduce_mean(grad,0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad,v)
        average_grads.append(grad_and_var)
    return average_grads

def train(dataset):
  """Train on dataset for a number of steps."""
  # with tf.Graph().as_default(), tf.device('/cpu:0'):
  with tf.Graph().as_default():

    # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #     global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

    global_step = tf.Variable(0,trainable=False)
    # global_step = tf.contrib.framework.get_or_create_global_step()

    decay_steps = 7500
    LEARNING_RATE_DECAY_FACTOR=0.1
    INITIAL_LEARNING_RATE=0.000001

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    opt = tf.train.MomentumOptimizer(learning_rate=lr ,momentum=0.1)

    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    with tf.device('/cpu:0'):
      images, pitchs, yaws, rolls, names = image_processing.distorted_inputs(
        dataset,
        num_preprocess_threads=num_preprocess_threads)
    
    p = tf.expand_dims(pitchs,1)
    y = tf.expand_dims(yaws,1)
    r = tf.expand_dims(rolls,1)
    labels = tf.concat([p, y, r],1)

    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity = 2*FLAGS.num_gpus)

    tower_grads = []

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    image_batch, label_batch = batch_queue.dequeue()
                    loss = tower_loss(scope,image_batch,label_batch)

                    tf.get_variable_scope().reuse_variables()

                    grads = opt.compute_gradients(loss)

                    tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(0.9999,global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(apply_gradient_op, variable_averages_op)

    saver = tf.train.Saver(tf.global_variables())

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)

    tf.train.start_queue_runners(sess = sess)
    

    for step in np.arange(FLAGS.max_steps):
  
        _, loss_value= sess.run([train_op, loss])
               
        if step % 50 == 0:
            print('Step %d, train loss = %.2f'  %(step, loss_value))
            
        if step % 2000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
                

    
