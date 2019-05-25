# !/usr/local/lib/python2.7 
# -*- coding=utf-8 -*-  

# bridging the python 2 and python 3 gap

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf # machine learning

# tf slim to define complex networks easily & quickly
slim = tf.contrib.slim 

## Multilayer neural networks

# Generative network
def Generator_AHB(y, input_dim, n_layer, n_hidden, eps_dim, reuse_):     
    with tf.variable_scope("generative",reuse=reuse_):         
        h = y               
        #many fully connected layers         
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)         
        x = slim.fully_connected(h, input_dim, activation_fn=None, scope="AHB")  
    return x

# Second generative network
def Generator_BHA(x, latent_dim, n_layer, n_hidden, eps_dim, reuse_):     
    with tf.variable_scope("inference",reuse=reuse_):         
        h = x         
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)         
        y = slim.fully_connected(h, latent_dim, activation_fn=None, scope="BHA")     
    return y 

# First discriminative network 
def Discriminator_AH(x, n_layers=2, n_hidden=10, activation_fn=None):
    """Approximate x log data density."""
    h = tf.concat(x, 1)
    with tf.variable_scope('discriminator_x'):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d)

# Second discriminative network
def Discriminator_BH(y, n_layers=2, n_hidden=10, activation_fn=None):
    """Approximate z log data density."""
    h = tf.concat(y, 1)
    with tf.variable_scope('discriminator_z'):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d)
