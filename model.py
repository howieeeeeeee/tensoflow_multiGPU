# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


NUM_EPOCHS_PER_DECAY=350.0
LEARNING_RATE_DECAY_FACTOR=10
INITIAL_LEARNING_RATE=0.01
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=12118
BATCH_SIZE=96

def inference(images, isTraining):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    
    conv1 = tf.layers.conv2d(
        inputs=images,
        filters=50,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.relu,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0005),
        kernel_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer = tf.ones_initializer(),
        name='conv1')
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
        pool_size=[2,2],
        strides=2)
    
    dropout1 = tf.layers.dropout(
        inputs=pool1,rate=0.2, training=isTraining)

    conv2 = tf.layers.conv2d(
        inputs=dropout1,
        filters=100,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.relu,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0005),
        kernel_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer = tf.ones_initializer(),
        name='conv2')

    pool2 = tf.layers.max_pooling2d(inputs=conv2,
        pool_size=[2,2],
        strides=2)

    dropout2 = tf.layers.dropout(
        inputs=pool2,rate=0.2, training=isTraining)

    conv3 = tf.layers.conv2d(
        inputs=dropout2,
        filters=150,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.relu,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0005),
        kernel_initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer = tf.ones_initializer())

    pool3 = tf.layers.max_pooling2d(inputs=conv3,
        pool_size=[2,2],
        strides=2)

    dropout3 = tf.layers.dropout(
        inputs=pool3,rate=0.3, training=isTraining)

    flat = tf.reshape(dropout3,[-1,8*8*150])

    dense1 = tf.layers.dense(inputs=flat,units=300,
        activation=tf.nn.relu,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0005),
        kernel_initializer = tf.contrib.layers.xavier_initializer(),
        use_bias=False)

    dropout4 = tf.layers.dropout(
        inputs=dense1,rate=0.3, training=isTraining)

    dense2 = tf.layers.dense(inputs=dropout4,units=300,
        activation=tf.nn.relu,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0005),
        kernel_initializer = tf.contrib.layers.xavier_initializer(),
        use_bias=False)

    dropout5 = tf.layers.dropout(
        inputs=dense2,rate=0.3, training=isTraining)

    output = tf.layers.dense(inputs=dropout5,units=3,
      kernel_regularizer = tf.contrib.layers.l2_regularizer(0.0005),
      kernel_initializer = tf.contrib.layers.xavier_initializer(),
      use_bias=False)
    
    return output
   

#%%
def losses(output, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        squared_deltas = tf.square(output-labels)
        loss = tf.reduce_sum(squared_deltas)
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = loss + regularization_loss
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate, global_step):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    with tf.name_scope('optimizer'):
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate ,momentum=0.9)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)

    return train_op

