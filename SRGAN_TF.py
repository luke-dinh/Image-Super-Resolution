import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)
from tensorlayer.layers import PRelu
from tensorlayer.models import Model

def Generator(input_shape):
    w_init = tf.random_normal_initializer(stddev = 0.02)
    g_init = tf.random_normal_initializer(1., 0.02)


    layer_in = Input(input_shape)
    l = Conv2d(64, (3,3), (1,1), padding = 'SAME', act = tf.nn.relu, w_init = w_init)(layer_in)
    temp = l

    #Residual Blocks
    for i in range(16):
        nn = Conv2d(64, (3,3), (1,1), act = tf.nn.relu, padding='SAME', W_init = w_init, b_init=None)(l)
        nn = BatchNorm2d(act = tf.nn.relu, gamma_init = g_init )(nn)
        #nn = PRelu(a_init = w_init)(nn)
        nn = Conv2d(64, (3,3), (1,1), act = tf.nn.relu, padding='SAME', W_init = w_init, b_init= None)(nn)
        nn = BatchNorm2d(act = tf.nn.relu, gamma_init = g_init)(nn)
        nn = Elementwise(tf.add)[l,nn]
        l = nn

    l = Conv2d(64, (3,3), (1,1), padding='SAME', W_init = w_init, b_init = None)(l)
    l = BatchNorm2d(gamma_init = g_init)(l)
    l = Elementwise(tf.add)[l, temp]

    l = Conv2d(256, (3,3), (1,1), padding='SAME', W_init = w_init)(l)
    l = SubpixelConv2d(scale = 2, n_out_channels=None, act = tf.nn.relu )(l)

    l = Conv2d(256, (3,3), (1,1), padding = 'SAME', W_init= w_init)(l)
    l = SubpixelConv2d(scale = 2, n_out_chanels = None, act = tf.nn.relu)(l)

    layer_out = Conv2d(3, (1,1), (1,1), act = tf.nn.tanh, padding = 'SAME', W_init= w_init)(l)
    Generator = Model(inputs = layer_in, outputs = layer_out, name = 'generator' )

    return Generator

def Discriminator(input_shape):
    w_init = tf.random_normal_initializer(stddev = 0.02)
    gamma_init = tf.random_normal_initializer(1.,0.02)
    dir_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    layer_in = Input(input_shape)

    l = Conv2d(dir_dim, (4,4), (2,2), act = lrelu, padding='SAME',W_init= w_init )(layer_in)
    l = BatchNorm2d(act = lrelu, gamma_init = gamma_init)(l)

    l = Conv2d(dir_dim * 2, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(l)
    l = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(l)
    l = Conv2d(dir_dim * 4, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(l)
    l = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(l)
    l = Conv2d(dir_dim * 8, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(l)
    l = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(l)
    l = Conv2d(dir_dim * 16, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(l)
    l = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(l)
    l = Conv2d(dir_dim * 32, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(l)
    l = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(l)
    l = Conv2d(dir_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(l)
    l = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(l)
    l = Conv2d(dir_dim * 8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(l)
    nn = BatchNorm2d(gamma_init=gamma_init)(l)

    l = Conv2d(dir_dim * 2, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
    l = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(l)
    l = Conv2d(dir_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(l)
    l = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(l)
    l = Conv2d(dir_dim * 8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(l)
    l = BatchNorm2d(gamma_init=gamma_init)(l)
    l = Elementwise(combine_fn=tf.add, act=lrelu)([l, nn])

    l = Flatten()(l)
    no = Dense(n_units=1, W_init=w_init)(l)
    Discriminator = Model(inputs=nin, outputs=no, name="discriminator")
    return Discriminator




