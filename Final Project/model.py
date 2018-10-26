#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:04:56 2018

@author: bao
"""

import tensorflow as tf
import numpy as np


class Model():
    def __init__(self,path):
        # Build the neural networks
        self.xs = tf.placeholder(tf.float32, [None, 28,28,1])/255.   # 28x28
        
        self.training = tf.placeholder(tf.bool)

        self.l1 = tf.layers.conv2d(inputs = self.xs,filters=32,kernel_size=(5,5),padding='same',activation=tf.nn.relu)

        self.l2 = tf.layers.conv2d(inputs = self.l1,filters=32,kernel_size=(5,5),padding='same',activation=tf.nn.relu)

        self.l3 = tf.layers.max_pooling2d(self.l2,pool_size=(2,2),strides=(1,1))
        
        self.l4 = tf.layers.batch_normalization(self.l3,training=self.training)

        self.l5 = tf.layers.dropout(self.l4,rate=0.25,training=self.training)

        self.l6 = tf.layers.conv2d(inputs = self.l5,filters=64,kernel_size=(3,3),padding='same',activation=tf.nn.relu)

        self.l7 = tf.layers.conv2d(inputs = self.l6,filters=64,kernel_size=(3,3),padding='same',activation=tf.nn.relu)

        self.l8 = tf.layers.max_pooling2d(self.l7,pool_size=(2,2),strides=(2,2))
        
        self.l9 = tf.layers.batch_normalization(self.l8,training=self.training)

        self.l10 = tf.layers.dropout(self.l9,rate=0.25,training=self.training)

        self.l11 = tf.layers.flatten(self.l10)

        self.l12 = tf.layers.dense(self.l11,256,activation = tf.nn.relu)
        
        self.l13 = tf.layers.batch_normalization(self.l12,training=self.training)

        self.l14 = tf.layers.dropout(self.l13,rate=0.5,training=self.training)

        self.l15 = tf.layers.dense(self.l14,9)

        self.predict = tf.nn.softmax(self.l15,dim = 1)
        
        self.sess = tf.Session()
       
        saver = tf.train.Saver()

        saver.restore(self.sess, path)
        
        print ("model loaded.")
    def classify(self,train_data):
        # Put the test data into the neural networks and return the result
        train_data = train_data.reshape(-1,28,28,1)
       
        result = self.predict.eval(session=self.sess,feed_dict={self.xs: train_data, self.training: False})
       
        return np.argmax(result,axis = 1)

        
    
