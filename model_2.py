from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def _variable_on_cpu(name, shape, initializer, istrain):

    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype = dtype, trainable=istrain)
    return var
def _variable_with_weight_decay(name, shape, istrain):

    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.random_normal_initializer(mean = 0,stddev=0.01,dtype=dtype),
        istrain)

    weight_decay = tf.multiply(tf.nn.l2_loss(var),0.0005, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var

def _variable_with_weight_decay_fc(name, shape, istrain):

    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.xavier_initializer(),
        istrain)

    weight_decay = tf.multiply(tf.nn.l2_loss(var), 0.0005, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var

def inference(images, istraining):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[5,5,3,50],
                                            istrain = True)
        conv = tf.nn.conv2d(images, kernel,[1,1,1,1], padding='VALID')
        biases = _variable_on_cpu('biases',[50], tf.constant_initializer(1.0),False)
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1],
                            padding='VALID', name='pool1')
    
    if istraining:
        dropout1 = tf.nn.dropout(pool1,0.2)
    else:
        dropout1 = tf.nn.dropout(pool1,1)
    # conv2 
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[5,5,50,100],
                                            istrain = True)
        conv = tf.nn.conv2d(dropout1, kernel, [1,1,1,1], padding='VALID')
        biases = _variable_on_cpu('biases',[100], tf.constant_initializer(1.0),False)
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
   
    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1],
                            padding='VALID', name='pool2')
    
    if istraining:
        dropout2 = tf.nn.dropout(pool2,0.2)
    else:
        dropout2 = tf.nn.dropout(pool2,0.2)
    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                            shape=[5,5,100,150],
                                            istrain = True)
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='VALID')
        biases = _variable_on_cpu('biases',[150], tf.constant_initializer(1.0),False)
        pre_activation = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    # pool3
    pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1],
                            padding='VALID', name='pool3')

    if istraining:
        dropout3 = tf.nn.dropout(pool3,0.3)
    else:
        dropout3 = tf.nn.dropout(pool3,1)
    

   # fc1
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(dropout3,[-1,8*8*150])
        weights = _variable_with_weight_decay_fc('weights',shape=[8*8*150,300],istrain=True)
        fc1 = tf.nn.relu(tf.matmul(reshape,weights),name=scope.name)

    if istraining:
        dropout4 = tf.nn.dropout(fc1,0.3)
    else:
        dropout4 = tf.nn.dropout(fc1,0.3)

    # fc2
    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay_fc('weights',shape=[300,300],istrain=True)
        fc2 = tf.nn.relu(tf.matmul(fc1,weights),name=scope.name)

    if istraining:
        dropout5 = tf.nn.dropout(fc2,0.3)  
    else:
        dropout5 = tf.nn.dropout(fc2,1)  

    with tf.variable_scope('output') as scope:
        weights = _variable_with_weight_decay_fc('weights',shape=[300,3],istrain=True)
        output = tf.matmul(fc1,weights)

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
        square_loss = tf.reduce_sum(squared_deltas)
        
        tf.add_to_collection('losses',square_loss)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')
