# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 17:47:52 2018

@author: bxc
"""
import gzip  
import os  
import tempfile  
  
import numpy  
from six.moves import urllib  
from six.moves import xrange  # pylint: disable=redefined-builtin  
import tensorflow as tf  
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets  
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import scipy.ndimage.interpolation
import random
from tensorflow.examples.tutorials.mnist import input_data

  
# Give a function to zoom the image but keeps the size after zooming.
# If the image is larger than 28*28 then crop it
# If the image is smaller than the 28*28 than padding it
def crop(original,H):
    h = original.shape[0]
    w = original.shape[1]
    if h>=H and w>=H:
        return original[int(h/2)-int(0.5*H):int(h/2)+int(0.5*H),int(w/2)-int(0.5*H):int(h/2)+int(0.5*H)]
    if h<H and w<H:
        new = np.zeros([H,H],dtype=original.dtype)
        new[int(0.5*H)-int(h/2):int(0.5*H)+int(h/2),int(0.5*H)-int(w/2):int(0.5*H)+int(w/2)]=original[0:2*int(h/2),0:2*int(w/2)]
        return new

def my_crop(image,H):
    re = crop(image.real,H)
    return re

def my_zoom(image,zoom):
    re = scipy.ndimage.zoom(image.real,zoom)
    return re
    
def my_shift(image,sdxy):
    re = scipy.ndimage.interpolation.shift(image.real,sdxy)
    return re

def my_rotate(image, angles, reshape):
    re = scipy.ndimage.interpolation.rotate(image, angles, reshape = False)
    return re

def deal_data_sets(path, rate=10):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_images = np.vstack([mnist.train.images,mnist.validation.images])
    train_labels = np.vstack([mnist.train.labels,mnist.validation.labels])
    # Change the label 9 to 6
    index = np.where(train_labels[:,-1] ==1)
    train_labels[:,-1]=0
    train_labels[index,-4]=1
    train_labels = train_labels[:,:-1]
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    index = np.where(test_labels[:,-1] ==1)
    test_labels[:,-1]=0
    test_labels[index,-4]=1
    test_labels = test_labels[:,:-1]
    data = []
    labels = []
    # Add some random data and save to mat
    for i in range(60000):
        for j in range(rate):
            print (i)
            data.append(random_data(train_images[i].reshape(28,28)))
            labels.append(train_labels[i])

    data = np.vstack([train_images.reshape(-1,28,28),np.array(data).reshape(-1,28,28)])
    labels = list(train_labels) + labels
    scipy.io.savemat("data_train.mat",{"images":np.array(data),"labels":np.array(labels)})    
            
    data = []
    labels = []

    for i in range(10000):
        for j in range(rate):
            print (i)
            data.append(random_data(test_images[i].reshape(28,28)))
            labels.append(test_labels[i])

    data = np.vstack([test_images.reshape(-1,28,28),np.array(data).reshape(-1,28,28)])
    labels = list(test_labels) + labels

    scipy.io.savemat("data_test.mat",{"images":np.array(data),"labels":np.array(labels)})  
    
def random_data(image,sdx=None,sdy=None,angles=None,zoom=None,trax=None,tray=None):
    #if sdx is None:
    #    sdx = random.randint(-3,3)
    #if sdy is None:
    #   sdy = random.randint(-3,3)
    if angles is None:
        angles = random.randint(0,360)
    if zoom is None:
        zoom = random.uniform(0.7,1.0)
    #if trax is None:
    #    trax = random.randint(0,1)
    #if tray is None:
    #    tray = random.randint(0,1)
    #image = my_shift(image,(sdx,sdy))
    image = my_rotate(image, angles, reshape = False)

    H = image.shape[0]
    image = crop(my_zoom(image, zoom),H=H)

    #if trax == 1:
    #    image = image[::-1,:]
    #if tray == 1:
    #    image = image[:,::-1]
    return image
