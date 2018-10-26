#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:58:57 2018

@author: bao
"""

'''
Comments about how to use batch_normalization found on the Internet

To add yet another alternative: as of TensorFlow 1.0 (February 2017) there's also the high-level tf.layers.batch_normalization API included in TensorFlow itself.

It's super simple to use:

# Set this to True for training and False for testing
training = tf.placeholder(tf.bool)

x = tf.layers.dense(input_x, units=100)
x = tf.layers.batch_normalization(x, training=training)
x = tf.nn.relu(x)

...except that it adds extra ops to the graph (for updating its mean and variance variables) in such a way that they won't be dependencies of your training op. You can either just run the ops separately:

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
sess.run([train_op, extra_update_ops], ...)

or add the update ops as dependencies of your training op manually, then just run your training op as normal:

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_op = optimizer.minimize(loss)
...
sess.run([train_op], ...)



'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


# Read the dataset
data_train = scipy.io.loadmat("data_train.mat")
data_test = scipy.io.loadmat("data_test.mat")

train_images = data_train["images"]

train_images = train_images.reshape(*train_images.shape,1)

train_labels = data_train["labels"]

test_images = data_test["images"]

test_images = test_images.reshape(*test_images.shape,1)

test_labels = data_test["labels"]

print (train_images.shape)
print (test_images.shape)


print (train_labels.shape)
print (test_labels.shape)

xs = tf.placeholder(tf.float32, [None, 28,28,1])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 9])
training = tf.placeholder(tf.bool)

# Build a model
l1 = tf.layers.conv2d(inputs = xs,filters=32,kernel_size=(5,5),padding='same',activation=tf.nn.relu)

l2= tf.layers.conv2d(inputs = l1,filters=32,kernel_size=(5,5),padding='same',activation=tf.nn.relu)

l3 = tf.layers.max_pooling2d(l2,pool_size=(2,2),strides=(1,1))

l4 = tf.layers.batch_normalization(l3,training=training)

l5 = tf.layers.dropout(l4,rate=0.25,training=training)

l6= tf.layers.conv2d(inputs = l5,filters=64,kernel_size=(3,3),padding='same',activation=tf.nn.relu)

l7= tf.layers.conv2d(inputs = l6,filters=64,kernel_size=(3,3),padding='same',activation=tf.nn.relu)

l8 = tf.layers.max_pooling2d(l7,pool_size=(2,2),strides=(2,2))

l9 = tf.layers.batch_normalization(l8,training=training)

l10 = tf.layers.dropout(l9,rate=0.25,training=training)

l11 = tf.layers.flatten(l10)

l12 = tf.layers.dense(l11,256,activation = tf.nn.relu)

l13 = tf.layers.batch_normalization(l12,training=training)

l14 = tf.layers.dropout(l13,rate=0.5,training=training)

l15 = tf.layers.dense(l14,9)

predict = tf.nn.softmax(l15,dim = 1)

loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=l15)

correct_prediction = tf.equal(tf.argmax(ys,1), tf.argmax(predict,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Define the optimizer
#optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.001, epsilon=1e-08, decay=0.0).minimize(loss)
optimizer = tf.train.MomentumOptimizer(learning_rate = 0.001,momentum=0.8).minimize(loss)



sess = tf.Session()
# tf.initialize_all_variables() does not work if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()

N = 60000*11
N_test = 10000*11
batch_size = 86
T = N // batch_size
epochs = 10

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

for epoch in range(epochs):
    for i in range(T):
        index = np.random.choice(N,batch_size)
        batch_xs, batch_ys = train_images[index,:,:,:],train_labels[index]
        _,_,loss_value = sess.run((optimizer,extra_update_ops,loss), feed_dict={xs: batch_xs, ys: batch_ys, training: True})
        if i%100 == 0:
            print ("epoch and step")
            print (epoch,i)
            print ("loss is:")
            print (loss_value)

            # Get the accurate rate for training data
            accuracy_value = accuracy.eval(session=sess,feed_dict={xs: train_images[index,:,:,:], ys: train_labels[index], training: False})
            print ("train error rate")
            print (1-accuracy_value)

            # Get the accurate rate for test data            
            index = np.random.choice(N_test,1000)
            accuracy_value = accuracy.eval(session=sess,feed_dict={xs: test_images[index,:,:,:], ys: test_labels[index], training: False})
            print ("error rate")
            print (1-accuracy_value)
            
# Save the model
saver.save(sess, "./model.ckpt")
