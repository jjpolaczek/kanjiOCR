# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

import time



#definitions from TF tutorial making code cleaner or sth
def weight_variable(shape):
    initial= tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
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
def batch_train(current, no_samples, dataset, labelCount):
    dataset_labels = dataset['train_labels']
    dataset_samples = dataset['train_dataset']
    charmap = dataset['label_map']
    if dataset_samples.shape[0] <= current + no_samples:
        no_samples = dataset_samples.shape[0] - current - 1
    data = dataset_samples[current:(current + no_samples)]
    # one hot encoding
    labels = np.zeros((no_samples,labelCount),dtype=np.float32)
    for i in range(no_samples):
       c = dataset_labels[current+i]
       labels[i][charmap[dataset_labels[current+i]]] = 1
    return data,labels
def test_data_get(dataset, labelCount):
    data = dataset['test_dataset'][:]
    charmap = dataset['label_map']
    labels = np.zeros((dataset['test_labels'].shape[0],labelCount),dtype=np.float32)
    for i in range(dataset['test_labels'].shape[0]):
       labels[i][charmap[dataset['test_labels'][i]]] = 1
    return data, labels



print('Training of M7.1 type neural network')

#dataset parameters
dataset = pickle.load(open("./data/ETL1.pickle", "rb"))
dimx = 75
dimy = 75
labelCount = len(dataset['label_map'])
trainSamples = dataset['train_labels'].shape[0]
testSamples = dataset['test_labels'].shape[0]
dataset['train_dataset'] = dataset['train_dataset'].astype(np.float32)
dataset['test_dataset'] = dataset['test_dataset'].astype(np.float32)
print ("Training %d x %d images, %d labels" % (dimx, dimy, labelCount))
print ("Training samples count - %d, test samples %d" % (trainSamples, testSamples))
print (dataset['train_dataset'].dtype)
#training parameters
batchSize = 100
restoreModel = False
nTrain = 5
#declare variables and io data
X = tf.placeholder(tf.float32, [None,75,75])
phase_train = tf.placeholder(tf.bool, name='phase_train')
#Xf = tf.reshape(X,[-1,75*75])
x_image = tf.expand_dims(X,3)#tf.reshape(Xf, [-1,75,75,1])
#first convolution layer and pooling operation
W_conv1 = weight_variable([3,3,1,64])
b_conv1 = bias_variable([64])
bn_conv1 = batch_norm(conv2d(x_image, W_conv1) + b_conv1,64, phase_train)
h_conv1 = tf.nn.relu(bn_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#second convolution layer
W_conv2 = weight_variable([3,3,64,128])
b_conv2 = bias_variable([128])
bn_conv2 = batch_norm(conv2d(h_pool1, W_conv2) + b_conv2,128,phase_train)
h_conv2 = tf.nn.relu(bn_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#third convolution layer
W_conv3 = weight_variable([3,3,128,512])
b_conv3 = bias_variable([512])
bn_conv3 = batch_norm(conv2d(h_pool2, W_conv3) + b_conv3,512,phase_train)
h_conv3 = tf.nn.relu(bn_conv3)
h_pool3 = max_pool_2x2(h_conv3)
#initialize fully connected layer 1 and flatten it
W_fc1 = weight_variable([10*10 * 512, 4096])
b_fc1 = bias_variable([4096])
h_pool3_flat = tf.reshape(h_pool3, [-1, 10*10*512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
# introduce dropout and keep rate svariable
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#fully connected layer 2
W_fc2 = weight_variable([4096, 4096])
b_fc2 = bias_variable([4096])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
#Readout layer
W_fc3 = weight_variable([4096,labelCount])
b_fc3 = bias_variable([labelCount])

Y = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

model_saver = tf.train.Saver()

Y_ = tf.placeholder(tf.float32, [None, labelCount])
#must write one hot encoded values to Y_
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y_,logits=Y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

currentIndex = 0

init = tf.global_variables_initializer()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
if restoreModel:
    model_saver.restore(sess, "CNN12.ckpt")
    print("Restoring state")
plot_it = []
plot_trainacc = []
#fig1 = plt.plot([],[])
plt.axis([0, (trainSamples / batchSize) * nTrain,0.8,1])
plt.ion()
plt.show()
timeStart = time.time()
for i in range((trainSamples / batchSize) * nTrain):

    #load and loop train images
    batch_X,  batch_Y = batch_train(currentIndex,batchSize,dataset, labelCount)
    if batch_X.shape[0] == 0:
        currentIndex = 0
        batch_X,  batch_Y = batch_train(currentIndex,batchSize,dataset, labelCount)
    currentIndex += batch_X.shape[0]
    #load train images and labelsinto tf session
    train_data={X: batch_X, Y_: batch_Y, keep_prob: 0.5, phase_train: True}
    #run optimizer defined previously on training data
    sess.run(train_step, feed_dict=train_data)
    #calculate accuracy and cross enthropy for training data
    a,c = sess.run([accuracy,cross_entropy],feed_dict=train_data)

    #fig1.set_xdata(numpy.array(plot_it,dtype=float32))
    #fig1.set_ydata(numpy.array(plot_trainacc,dtype=int))
    #assess performance on test data
    if i%(100 / (batchSize / 100)) == 0:
        #perform test on test dataset
        plt.pause(0.01)
        testA, testB = test_data_get(dataset, labelCount)
        resA = []
        for j in range(len(testA) / 100):
            chunkA = testA[100*j:100*(j+1)]
            chunkB = testB[100*j:100*(j+1)]
            a = accuracy.eval(feed_dict={X: chunkA, Y_: chunkB, keep_prob: 1.0, phase_train: False},session=sess)
            resA.append(a)
        a = 0
        for j in range(len(resA)):
            a += resA[j]
        a /= len(resA)
        
        #plot data

        plot_trainacc.append(a)
        plot_it.append(i)
        plt.plot(plot_it,plot_trainacc)
        print('accuracy: ',a)
        timeStop = time.time()
        print("Time elapsed: ", timeStop - timeStart)
        timeStart = timeStop
    if i %(2000 / (batchSize / 100)) == 0:
        model_saver.save(sess, "CNN12.ckpt")
        print("Saving state")
    
#optimize

