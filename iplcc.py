###################################################################################################
#--------------------------------------------------------------------------------------------------
#
#                                  Michael A. Teti
# 
#                Machine Perception and Cognitive Robotics Laboratory
# 
#                   Center for Complex Systems and Brain Sciences 
#
#                          Florida Atlantic University
#
###################################################################################################
#--------------------------------------------------------------------------------------------------
###################################################################################################
#
# This convolutional neural network was developed to classify pixels contained in remotely-sensed
# hyperspectral images obtained from the following source:
# 
# http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
#
#--------------------------------------------------------------------------------------------------
###################################################################################################


import scipy
from scipy.io import loadmat
import scipy.misc
import numpy as np
import os 
import sys

from __future__ import print_function
import tensorflow as tf

import TensorflowUtils as utils
import DataParser as scene_parsing
import datetime
import BatchDatasetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("gpu", "0", "which gpu to use")
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("cpkt_dir", "./checkpoints/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "./data/match/", "path to dataset")
tf.flags.DEFINE_string("preds_dir", "./data/match/preds/", "path to output predictions")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model/", "Path to model mat")
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize/ infer")


# load data and ground truth
a=loadmat('Pavia.mat') # .mat file that has the data
b=loadmat('Pavia_gt.mat') # load .mat file with the ground truth
gt=b['pavia_gt']
ps=4 
gt=np.pad(gt, (ps, ps), 'edge') # pad ground truth to coincide with data
data=a['pavia']
depth=np.ma.size(data, 2) # number of bands used in sensing
data=np.pad(data, (ps, ps), 'edge') # pad the data matrix to select patches
output=np.ndarray.max(gt) # 9 output nodes in CNN for 9 classes in scene
batch=1000 # training batch size
c1=65 # convolutions in layer 1
c2=70 # convolutions in layer 2
h1=np.round((4*c2/2), decimals=0) # nodes in 1st fully-connected layer

sess=tf.InteractiveSession() # begin tensorflow interactive session


x = tf.placeholder(tf.float32, shape=[None, depth*49]) # input layer of nodes to send data through
y = tf.placeholder(tf.float32, shape=[None, output]) # output layer of the network
keep_prob = tf.placeholder(tf.float32) # dropout probability placeholder

# create the training batch
def training_set(X, y):
	td=np.empty([1, depth*49])
	labels=np.zeros([batch, output])
	for i in range(batch):
		train_pixelr, train_pixelc=np.nonzero(y) # find nonzero elements in ground truth (zeros correspond to no class)
		randp=np.random.permutation(np.ma.size(train_pixelr)) # choose a random selection of these elements
		train_pixelr=train_pixelr[randp[0]]
		train_pixelc=train_pixelc[randp[0]]
		# Don't select padded elements - leave space for patch size
		if train_pixelr<ps: 
			train_pixelr=ps
		elif train_pixelr>(np.ma.size(y, 0)-ps):
			train_pixelr=np.ma.size(y, 0)-ps
		if train_pixelc<ps:
			train_pixelc=ps
		elif train_pixelc>(np.ma.size(y, 1)-ps):
			train_pixelc=np.ma.size(y, 1)-ps
		# select 7x7 patches with the pixel to be classified in the center
		train_data=X[train_pixelr-(ps-1):train_pixelr+ps, train_pixelc-(ps-1):train_pixelc+ps, ps:depth+ps]
		train_data=train_data.flatten()
		train_data=train_data[np.newaxis, :]
		td=np.concatenate((td, train_data), axis=0)
		label=y[train_pixelr, train_pixelc]
		labels[i, label-1]=1 # create label matrix from ground truth
	td=td[1:, :]
	# feature scaling of the data
	mu=np.mean(td, axis=0)
	sigma=np.std(td, axis=0)
	td=np.divide((td-mu), sigma)
	td=np.nan_to_num(td)
	if np.ma.size(td, 0)!=batch:
		print 'Data not loaded correctly. Goodbye'
		sys.exit()
	return td, labels


# create small, random weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# create biases
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# first convolutional layer
X=tf.reshape(x, [-1, 7, 7, depth]) # reshape training example into 7x7x103 patches
w1=weight_variable([2, 2, depth, c1]) # first convolutional layer is 1x3
bias1=bias_variable([c1])
act1=max_pool_2x2(tf.nn.elu(conv2d(X, w1)+bias1)) # apply convolutions, rectified linear activation function, and max pooling

w2=weight_variable([2, 2, c1, c2]) # second convolutional layer is 3x1
bias2=bias_variable([c2])
act2=max_pool_2x2(tf.nn.elu(conv2d(act1, w2)+bias2))
act2flat=tf.reshape(act2, [-1, 4*c2]) # reshape layer 2 activations into a vector for fully-connected layer

theta1=weight_variable([4*c2, h1]) # weights for fully-connected layer
bias3=bias_variable([h1]) # bias for first fully-connected layer
activation2=tf.nn.elu(tf.matmul(act2flat, theta1) + bias3) # activations for fully-connected layer
activation2 = tf.nn.dropout(activation2, keep_prob) # apply dropout to activations to avoid overfitting

theta2=weight_variable([h1, h1]) # weights for second fully-connected layer
bias4=bias_variable([h1]) # bias for 2nd layer
activation3=tf.nn.elu(tf.matmul(activation2, theta2) + bias4) # multiply activations from layer one by weights and add bias
activation3=tf.nn.dropout(activation3, keep_prob) # apply dropout to this layer of activations too

theta3=weight_variable([h1, output]) # weights for final layer of the network (output layer)
bias5=bias_variable([output]) # biases for the output nodes
out=tf.nn.softmax(tf.matmul(activation3, theta3) + bias5) # send through a softmax regression function 


# calculate cross-entropy cost function to use with backpropagation
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))

# backpropagation optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

# return one if the output node for a training example is the same as the label
correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))

# calculate the mean for the training batch
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver()

sess.run(tf.initialize_all_variables()) # initialize the tensorflow variables

training_iters=525

# training
for i in range(training_iters):
  train_data, train_labels=training_set(data, gt)
  if i%5 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:train_data, y:train_labels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    #tf.Print(y_, train_labels)
  train_step.run(feed_dict={x: train_data, y: train_labels, keep_prob: 0.5})

print ('Saving variables...')
save_variables=saver.save(sess, '/home/mpcr/Desktop/Indian_pines/hyperspectral.ckpt')
