# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import os
import sys
import tensorflow as tf
import time
from six.moves import cPickle as pickle
import cv2

#definitions from TF tutorial making code cleaner or sth
def weight_variable(shape, vname):
    initial= tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=vname)
def bias_variable(shape,vname):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=vname)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding='SAME')
def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
class KanjiNN:
    def __init__(self,pathToModel, pathToDict):
        print ("Loading Dictionary")
        self.dictionary = pickle.load(open(pathToDict, "rb"))
        print ("Load Complete")
        labelCount = len(self.dictionary)
        print ("Detected %d character labels" %(labelCount))
        
        #declare variables and io data
        self.X = tf.placeholder(tf.float32, [None,75,75])
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        #Xf = tf.reshape(X,[-1,75*75])
        x_image = tf.expand_dims(self.X,3)#tf.reshape(Xf, [-1,75,75,1])
        #first convolution layer and pooling operation
        W_conv1 = weight_variable([3,3,1,64], 'W_conv1')
        b_conv1 = bias_variable([64], 'b_conv1')
        bn_conv1 = batch_norm(conv2d(x_image, W_conv1) + b_conv1,64, self.phase_train,scope='bn1')
        h_conv1 = tf.nn.relu(bn_conv1)
        #pooling 1
        h_pool1 = max_pool_2x2(h_conv1)
        #second convolution layer
        W_conv2 = weight_variable([3,3,64,128],'W_conv2')
        b_conv2 = bias_variable([128],'b_conv2')
        bn_conv2 = batch_norm(conv2d(h_pool1, W_conv2) + b_conv2,128,self.phase_train,scope='bn2')
        h_conv2 = tf.nn.relu(bn_conv2)
        #pooling 2
        h_pool2 = max_pool_2x2(h_conv2)
        #third convolution layer
        W_conv3 = weight_variable([3,3,128,512],'W_conv3')
        b_conv3 = bias_variable([512],'b_conv3')
        bn_conv3 = batch_norm(conv2d(h_pool2, W_conv3) + b_conv3,512,self.phase_train,scope='bn3')
        h_conv3 = tf.nn.relu(bn_conv3)
        #fourth convolution layer
        W_conv4 = weight_variable([3,3,512,512],'W_conv4')
        b_conv4 = bias_variable([512],'b_conv4')
        bn_conv4 = batch_norm(conv2d(h_conv3, W_conv4) + b_conv4,512,self.phase_train,scope='bn4')
        h_conv4 = tf.nn.relu(bn_conv4)
        #pooling 3
        h_pool4 = max_pool_2x2(h_conv4)
        #initialize fully connected layer 1 and flatten it
        W_fc1 = weight_variable([10*10 * 512, 4096],'W_fc1')
        b_fc1 = bias_variable([4096],'b_fc1')
        h_pool4_flat = tf.reshape(h_pool4, [-1, 10*10*512])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
        # introduce dropout and keep rate svariable
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        #fully connected layer 2
        W_fc2 = weight_variable([4096, 4096],'W_fc2')
        b_fc2 = bias_variable([4096],'b_fc2')
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)
        #Readout layer
        W_fc3 = weight_variable([4096,labelCount],'W_fc3')
        b_fc3 = bias_variable([labelCount],'b_fc3')

        self.Y = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

        Y_ = tf.placeholder(tf.float32, [None, labelCount])
        #must write one hot encoded values to Y_
    #    cross_entropy = tf.reduce_mean(
    #        tf.nn.softmax_cross_entropy_with_logits(labels=Y_,logits=Y))
    #    is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
    #    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        saver = tf.train.Saver()
        print("Restoring state")
        saver.restore(self.sess, pathToModel)
        print("Restore Complete")
    def ProcessImage(self,image):
        if image.shape != (75,75):
            print ("Invalid shape", image.shape)
            return None
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        #return 1
        #load train images and labelsinto tf session
        train_data={self.X: image, self.keep_prob: 1.0, self.phase_train: False}
        #calculate accuracy and cross enthropy fordata
        pred = self.sess.run(tf.argmax(self.Y,1),feed_dict=train_data)
        return(self.dictionary[pred[0]])
