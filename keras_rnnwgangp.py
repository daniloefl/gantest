#!/usr/bin/env python3

# to be able to run this:
# sudo apt-get install python3 python3-pip
# pip3 install --user matplotlib seaborn numpy tensorflow keras h5py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import gc

import matplotlib as mpl
mpl.use('Agg')

# numerical library
import numpy as np
# to manipulate data using DataFrame
import h5py

#import plaidml.keras
#import plaidml
#plaidml.keras.install_backend()

from keras.layers import Input, Dense, Layer, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import binary_crossentropy, mean_squared_error

import keras as K
from utils import LayerNormalization
import tensorflow as tf

mnist = K.datasets.mnist

def smoothen(y):
  return y
  N = 3
  box = np.ones(N)/float(N)
  return np.convolve(y, box, mode = 'same')

def gradient_penalty_loss(y_true, y_pred, critic, generator, z, real, n_x, n_y):
  Ns = K.backend.shape(real)[0]
  d1 = generator(z)
  d2 = real
  diff = d2 - d1
  epsilon = tf.random.uniform(shape=[Ns, 1, 1, 1], minval = 0., maxval = 1.)
  interp_input = d1 + (epsilon*diff)
  gradients = K.backend.gradients(critic(interp_input), [interp_input])[0] # shape = (None, 28, 28, 1)
  ## not needed as there is a single element in interp_input here (the discriminator output)
  ## the only dimension left is the batch, which we just average over in the last step
  slopes = K.backend.sqrt(1e-7 + K.backend.sum(K.backend.square(gradients), axis = [1, 2, 3]))
  gp = K.backend.mean(K.backend.square(1 - slopes))
  return gp

#def gradient_penalty_loss(y_true, y_pred, critic, generator, z, real, n_x, n_y):
#  Ns = K.backend.shape(real)[0]
#  d1 = generator(z)
#  d2 = real
#  diff = d2 - d1
#  epsilon = tf.random.uniform(shape=[Ns, 1, 1, 1], minval = 0., maxval = 1.)
#  interp_input = d1 + (epsilon*diff)
#  gradients = tf.reshape(tf.gradients(critic(interp_input), interp_input), [-1, n_x*n_y])
#  slopes = tf.sqrt(1e-6 + tf.reduce_sum(tf.square(gradients), axis = 1))
#  gp = tf.reduce_mean(tf.square(1 - slopes))
#  ## not needed as there is a single element in interp_input here (the discriminator output)
#  ## the only dimension left is the batch, which we just average over in the last step
#  ##grad_sum = tf.reduce_sum(tf.square(gradients), axis = 1) > 1.
#  ##grad_safe = tf.where(grad_sum, gradients, tf.ones_like(gradients))
#  ##grad_abs = 0. * gradients
#  ##grad_norm = tf.where(grad_sum, tf.norm(grad_safe, axis = 1), tf.reduce_sum(grad_abs, axis = 1))
#  ##gp = tf.reduce_mean(tf.square(grad_norm - 1.0))
#  return gp

def wasserstein_loss(y_true, y_pred):
  return K.backend.mean(y_true*y_pred, axis = 0)

class GenerateImage(K.layers.Layer):

    def __init__(self,
                 n_x = 28,
                 n_y = 28,
                 n_pix = 256,
                 **kwargs):
        """Generate image from positions
        :param n_x: Size in x.
        :param n_y: Size in y.
        :param n_pix: Number of pixels produced.
        :param kwargs:
        """
        super(GenerateImage, self).__init__(**kwargs)
        self.n_x = n_x
        self.n_y = n_y
        self.n_pix = n_pix

    def get_config(self):
        config = {
               'n_x': self.n_x,
               'n_y': self.n_y,
               'n_pix': self.n_pix,
        }
        base_config = super(GenerateImage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.n_x, self.n_y, 1)

    def build(self, input_shape):
        super(GenerateImage, self).build(input_shape)

    def call(self, inputs, training=None):
        # input shape for each input: (Nbatch, n_pix)

        ####### The following is to be used if the network predicts a single number with nx and ny positions, instead of giving a softmax output
        ####### if the pos_x and pos_y layers generate a single number (shape = (Nbatch, Npix, 1))
        ####### Begin -----
        ######pos_x = inputs[0]*self.n_x*0.5 + self.n_x*0.5
        ######pos_y = inputs[1]*self.n_y*0.5 + self.n_y*0.5
        ######energy = inputs[2]
        ####### get batch size
        ######Nbatch = K.backend.shape(energy)[0]
        ####### get number of pixels
        ######Npix = self.n_pix # K.backend.shape(energy)[1]

        ####### now we map the x and y positions to a single number given by x + Nx * y
        ####### afterwards we can just get the i-th item of an identity matrix, multiply it by the energy and sum

        ####### position after rolling out
        ######pos_rolled_out = pos_x + self.n_x*pos_y # results in a number from 0 to self.n_x*self.n_y of shape (Nbatch, Npix, 1)

        ####### this should work, but it is not differentiable
        ########pos_rolled_out = tf.cast(pos_rolled_out, tf.int32)
        ####### identity matrix used to fill each bin after rolling out
        ########identity = tf.eye(self.n_x*self.n_y)
        ####### these are rows with ones at the exact position
        ########pos_rows = tf.gather_nd(identity, pos_rolled_out) # shape is (Nbatch, Npix, nx*ny)

        ####### this should also work, but it is not differentiable
        ########pos_rolled_out = tf.cast(pos_rolled_out, tf.int32)
        ####### this is equivalent using one_hot
        ########pos_rows = tf.one_hot(pos_rolled_out, self.n_x*self.n_y)
        ########pos_rows = tf.reshape(pos_rows, [Nbatch, Npix, self.n_x*self.n_y])

        ####### this uses a differentiable approximation
        ####### generate the an axis with the number of pixels rolled out simply counting from 0 to the maximum number of pixels
        ####### it is kept with the first two axes so it can be expanded in Nbatch and Npix later
        ######x_range = tf.reshape(tf.range(self.n_x*self.n_y, dtype = tf.float32), [1, 1, self.n_x*self.n_y])

        ####### now create a Gaussian shape along the x_range axis, using the pos_rolled_out as a mean
        ####### since pos_rolled_out has shape (Nbatch, Npix, 1), the first two axes are expanded in subtraction, by tiling x_range
        ####### the standard deviation of this Gaussian is set to 1.0/10.0, so that it is smaller than the resolution in x_range
        ####### this effectively leads to a single entry in pos_rows to be 1 and all other entries to be negligible, as we expect from one_hot
        ####### note that the normalisation of the maximum is 1, because pos_rolled_out - x_range is zero at the correct position and exp(0) = 1
        ####### the small standard deviation takes care of leaving all other entries at small values
        ######pos_rows = tf.exp(-tf.square(pos_rolled_out - x_range)*0.5*(10.0**2.0))
        ####### the shape of pos_rows should be (Nbatch, Npix, nx*ny)

        ####### reshape the energy to repeat its value nx*ny times in a new dimension
        ######energy_tiled = tf.tile(energy, [1, 1, self.n_x*self.n_y])
        ####### energy_tiled has shape [Nbatch, Npix, self.n_x*self.n_y] and it has the same value always in the last axis

        ####### multiplying pos_rows by energy_tiled leads to a tensor with shape (Nbatch, Npix, nx*ny), where only one entry in the last axis has a non-zero value
        ####### that value is exactly the energy of the pixel
        ####### after that, we can sum in the axis 1 (the one of length Npix), and this will produce a single array of size nx*ny per batch with the sum of energies
        ####### in each pixel corresponding to the position in the last axis

        ####### the image rolled out is the i-th row corresponding to the required position (will have 1 only in the i-th column), scaled by the energy and summed
        ####### the resulting image
        ######image_rolled_out = tf.reduce_sum(pos_rows * energy_tiled, axis = 1) # sums over pixels, as the shape of both is (Nbatch, Npix, nx*ny)
        ####### reduce_sum reduces the axis 1 of dimension Npix, so image_rolled_out has shape (Nbatch, nx*ny)

        ####### to make the image in its appropriate format, just reshape it
        ######image = tf.reshape(image_rolled_out, [Nbatch, self.n_x, self.n_y, 1])

        ####### End -----

        ## This is the new setup:
        ## if the pos_x and pos_y layers generate a softmax output (shape = (Nbatch, Npix, nx/y))
        ## Begin -----
        #pos_x = inputs[0]
        #pos_y = inputs[1]
        #energy = inputs[2]
        ## get batch size
        #Nbatch = K.backend.shape(energy)[0]
        ## get number of pixels
        #Npix = self.n_pix # K.backend.shape(energy)[1]

        #pos_x_tiled = tf.tile(tf.reshape(pos_x, [Nbatch, Npix, self.n_x, 1]), [1, 1, 1, self.n_y])
        #pos_y_tiled = tf.tile(tf.reshape(pos_y, [Nbatch, Npix, 1, self.n_y]), [1, 1, self.n_x, 1])
        #pos_rows = pos_x_tiled*pos_y_tiled

        ## now pos_rows has shape (Nbatch, Npix, nx, ny)
        ## we can scale it by energy to have one image with the correct energy per pixel (blurred out depending on how certain the network is to output a single pixel or many)
        #energy_tiled = tf.tile(tf.tile(tf.reshape(energy, [Nbatch, Npix, 1, 1]), [1, 1, self.n_x, 1]), [1, 1, 1, self.n_y])

        ## now sum all pixels
        #image = tf.reduce_sum(pos_rows * energy_tiled, axis = 1) # sums over pixels, as the shape of both is (Nbatch, Npix, nx, ny)
        #image = tf.reshape(image, [Nbatch, self.n_x, self.n_y, 1]) # add axis for layer

        ## End -----

        ## This is an even newer!!! setup:
        ## not using a softmax ... returning the exact single x and y
        ## pos_x and pos_y have shape (Nbatch, Npix, 1)
        ## Begin -----
        #energy = inputs[2]
        ## get batch size
        #Nbatch = K.backend.shape(energy)[0]
        ## get number of pixels
        #Npix = self.n_pix # K.backend.shape(energy)[1]

        ## not using a softmax ... returning the exact single x and y
        ## pos_x and pos_y have shape (Nbatch, Npix, 1)
        #pos_x = inputs[0]*(self.n_x-1)
        #pos_y = inputs[1]*(self.n_y-1)
        #x_range = tf.reshape(tf.range(self.n_x*self.n_y, dtype = tf.float32), [1, 1, self.n_x*self.n_y])
        #pos_rolled_out = pos_x + self.n_x*pos_y # results in a number from 0 to self.n_x*self.n_y of shape (Nbatch, Npix, 1)
        #pos_rolled_out = tf.tile(pos_rolled_out, [1, 1, self.n_x*self.n_y])
        #x_range = tf.tile(x_range, [Nbatch, 1, 1])
        #x_range = tf.tile(x_range, [1, Npix, 1])
        #pos_rows = tf.exp(-tf.square(pos_rolled_out - x_range)*0.5*(10.0**2.0))

        ## now pos_rows has shape (Nbatch, Npix, nx*ny)
        ## we can scale it by energy to have one image with the correct energy per pixel
        ## (blurred out depending on how certain the network is to output a single pixel or many)
        #energy_rolled_out = tf.tile(energy, [1, 1, self.n_x*self.n_y])

        ## now sum all pixels
        #image = tf.reduce_sum(pos_rows * energy_rolled_out, axis = 1) # sums over pixels, as the shape of both is (Nbatch, Npix, nx*ny)
        #image = tf.reshape(image, [Nbatch, self.n_x*self.n_y, 1]) # add axis for layer
        #image = tf.reshape(image, [Nbatch, self.n_x, self.n_y, 1]) # split axis 3 in x and y positions
        ## End -----

        # This is the new setup:
        # if the pos_xy generate a softmax output (shape = (Nbatch, Npix, nx*ny))
        # Begin -----
        pos_xy = inputs[0]
        energy = inputs[1]
        # get batch size
        Nbatch = K.backend.shape(energy)[0]
        # get number of pixels
        Npix = self.n_pix # K.backend.shape(energy)[1]

        energy_tiled = tf.tile(energy, [1, 1, self.n_x*self.n_y])

        # now sum all pixels
        image = tf.reduce_sum(pos_xy * energy_tiled, axis = 1) # sums over pixels, as the shape of both is (Nbatch, Npix, nx*ny)
        image = tf.reshape(image, [Nbatch, self.n_x, self.n_y, 1]) # add axis for layer

        # End -----

        return image


class RNNWGANGP(object):
  '''
  Implementation of the Wasserstein GAN with Gradient Penalty algorithm to generate samples of a training set.
  Ref.: https://arxiv.org/pdf/1704.00028.pdf
  Ref.: https://arxiv.org/abs/1701.07875
  Ref.: https://arxiv.org/abs/1406.2661
  The objective of the generator is to generate images (Y). The critic punishes the generator
  if it can guess whether the sample is real or fake.

  The generator outputs o = G(z), where z is a random normal sample.

  The critic estimates a function C(o), where o = G(z), such that the Wasserstein distance between real and fake images
  in the output of the discriminator can be measured as:
  W(real, fake) = E_{fake}[C(G(z))] - E_{real}[C(x)]

  One needs to impose that the function C(.) is Lipschitz, so that W(.) actually implements the Wasserstein distance, according to
  the Katorovich-Rubinstein duality. The original Wasserstein paper imposes it by restricting the critic network weights to be close to zero.
  Here, the gradient penalty is used, on which we impose that the norm of the gradient of the critic function is close to 1 everywhere
  in the line that connects nominal and uncertainty samples (it should be everywhere though.

  The following procedure is implemented:
  1) Train the critic, fixing the generator, in n_critic batches to minimize the Wasserstein distance
     between real and fake in the output of the generator, respecting the gradient penalty condition.
     epsilon = batch_size samples of a uniform distribution between 0 and 1
     o_{itp} = epsilon x_{real} + (1 - epsilon) x_{fake}
  L_{critic} = { E_{fake} [ C(G(z)) ] - E_{real} [ C(x) ] } + \lambda_{GP} [ 1 - || grad_{o_{itp}} C(o_{itp}) || ]^2

  2) Train the generator, fixing the critic, in one batch to
     move in the direction of - grad W = E [grad_{generator weights} C(G(z)) ] (Theorem 3 of the Wasserstein paper).
  L_{gen} = - E_{fake} [ C(G(z)) ]

  3) Go back to 1 and repeat this n_iteration times.
  '''

  def __init__(self, n_iteration = 30000, n_critic = 5,
               n_batch = 100,
               lambda_gp = 10.0,
               n_eval = 50,
               n_x = 28, n_y = 28,
               n_pix = 256,
               n_dimensions = 20):
    '''
    Initialise the network.

    :param n_iteration: Number of batches to run over in total.
    :param n_critic: Number of batches to train the critic on per batch of the discriminator.
    :param n_batch: Number of samples in a batch.
    :param lambda_gp: Lambda parameter to weight the gradient penalty of the critic loss function.
    :param n_eval: Number of batches to train before evaluating metrics.
    '''
    self.n_iteration = n_iteration
    self.n_critic = n_critic
    self.n_batch = n_batch
    self.lambda_gp = lambda_gp
    self.n_eval = n_eval
    self.critic = None
    self.generator = None
    self.n_x = n_x
    self.n_y = n_y
    self.n_pix = n_pix
    self.n_dimensions = n_dimensions

  '''
    Create critic network.
  '''
  def create_critic(self):
    self.critic_input = Input(shape = (self.n_x, self.n_y, 1), name = 'critic_input')

    xc = self.critic_input

    xc = K.layers.Conv2D(32, (3,3), padding = "same", activation = None, name = "adv_0")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None, name = "adv_1")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.MaxPooling2D(pool_size = (2, 2), name = "adv_2")(xc)
    #xc = K.layers.Dropout(0.5)(xc)

    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None, name = "adv_3")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None, name = "adv_4")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.MaxPooling2D(pool_size = (2, 2), name = "adv_5")(xc)
    #xc = K.layers.Dropout(0.5)(xc)

    xc = K.layers.Flatten()(xc)
    xc = K.layers.Dense(512, activation = None, name = "adv_6")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    #xc = K.layers.Dropout(0.5)(xc)
    xc = K.layers.Dense(128, activation = None, name = "adv_7")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Dense(64, activation = None, name = "adv_8")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Dense(1, activation = None, name = "adv_9")(xc)

    self.critic = Model(self.critic_input, xc, name = "critic")
    self.critic.trainable = True
    self.critic.compile(loss = wasserstein_loss,
                        optimizer = Adam(lr = 1e-4), metrics = [])

  '''
  Create generator network.
  https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
  '''
  def create_generator(self):
    self.generator_input = Input(shape = (None, self.n_dimensions,), name = 'generator_input')

    xg = self.generator_input
    xg = K.layers.recurrent.LSTM(256, return_sequences = True)(xg)
    #xg = K.layers.recurrent.LSTM(256, return_sequences = True)(xg)
    #xg = K.layers.recurrent.LSTM(128, return_sequences = True)(xg)
    xg = K.layers.TimeDistributed(K.layers.Dense(512, activation = None))(xg)
    xg = K.layers.TimeDistributed(K.layers.LeakyReLU(0.2))(xg)
    xg = K.layers.TimeDistributed(K.layers.Dense(256, activation = None))(xg)
    xg = K.layers.TimeDistributed(K.layers.LeakyReLU(0.2))(xg)
    xg = K.layers.TimeDistributed(K.layers.Dense(128, activation = None))(xg)
    xg = K.layers.TimeDistributed(K.layers.LeakyReLU(0.2))(xg)
    #pos_x = K.layers.TimeDistributed(K.layers.Dense(self.n_x, activation = 'softmax'))(xg)
    #pos_y = K.layers.TimeDistributed(K.layers.Dense(self.n_y, activation = 'softmax'))(xg)
    #pos_x = K.layers.TimeDistributed(K.layers.Dense(1, activation = 'sigmoid'))(xg)
    #pos_y = K.layers.TimeDistributed(K.layers.Dense(1, activation = 'sigmoid'))(xg)
    pos_xy = K.layers.TimeDistributed(K.layers.Dense(self.n_x*self.n_y, activation = 'softmax'))(xg)
    energy = K.layers.TimeDistributed(K.layers.Dense(1, activation = 'relu'))(xg)
    xg = GenerateImage(self.n_x, self.n_y, self.n_pix)([pos_xy, energy])

    #xg = K.layers.Dropout(0.5)(xg)

    self.generator = Model(self.generator_input, xg, name = "generator")
    self.generator.trainable = True
    self.generator.compile(loss = K.losses.mean_squared_error, optimizer = Adam(lr = 1e-4), metrics = [])
    self.generator.summary()

  '''
  Create all networks.
  '''
  def create_networks(self):
    if not self.critic:
      self.create_critic()
    if not self.generator:
      self.create_generator()

    self.generator.trainable = False
    self.critic.trainable = True

    self.dummy_input = Input(shape = (1,), name = 'dummy_input')
    self.z_input = Input(shape = (None, self.n_dimensions,), name = 'z_input')
    self.real_input = Input(shape = (self.n_x, self.n_y, 1), name = 'real_input')

    from functools import partial
    partial_gp_loss = partial(gradient_penalty_loss, critic = self.critic, generator = self.generator, z = self.z_input, real = self.real_input, n_x = self.n_x, n_y = self.n_y)

    wdistance = K.layers.Subtract()([self.critic(self.generator(self.z_input)),
                                     self.critic(self.real_input)])

    self.gen_fixed_critic = Model([self.real_input, self.z_input, self.dummy_input],
                                   [wdistance, self.dummy_input],
                                   name = "gen_fixed_critic")
    self.gen_fixed_critic.compile(loss = [wasserstein_loss, partial_gp_loss],
                                   loss_weights = [1.0, self.lambda_gp],
                                   optimizer = Adam(lr = 1e-4), metrics = [])

    self.generator.trainable = True
    self.critic.trainable = False
    self.gen_critic_fixed = Model([self.z_input],
                                   [self.critic(self.generator(self.z_input))],
                                   name = "gen_critic_fixed")
    self.gen_critic_fixed.compile(loss = [wasserstein_loss],
                                   loss_weights = [-1.0],
                                   optimizer = Adam(lr = 1e-4), metrics = [])


    print("Generator:")
    self.generator.trainable = True
    self.generator.summary()
    print("Critic:")
    self.critic.trainable = True
    self.critic.summary()
    print("Gen. against critic:")
    self.generator.trainable = True
    self.critic.trainable = False
    self.gen_critic_fixed.summary()
    print("Critic against gen.:")
    self.generator.trainable = False
    self.critic.trainable = True
    self.gen_fixed_critic.summary()


  '''
  '''
  def read_input_from_files(self):
    (self.x_train, self.y_train), (self.x_test, self.y_test) = K.datasets.mnist.load_data()
    self.x_train = self.x_train.astype('float32')
    self.x_test = self.x_test.astype('float32')
    self.x_train /= 255.0
    self.x_test /= 255.0

  def plot_data(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize = (20, 20), nrows = 5, ncols = 5)
    out = self.get_batch(origin = 'train', size = 5*5)
    for i in range(5):
      for j in range(5):
        sns.heatmap(out[j+5*i,:,:,0], vmax = .8, square = True, ax = ax[i, j])
        ax[i, j].set(xlabel = '', ylabel = '', title = '');
    plt.savefig(filename)
    plt.close("all")

  def plot_generator_output(self, filename, network_batch = 0):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize = (20, 20), nrows = 5, ncols = 5)
    z = np.random.normal(loc = 0.0, scale = 1.0, size = (5*5, self.n_pix, self.n_dimensions,))
    out = self.generator.predict(z, verbose = 0)
    for i in range(5):
      for j in range(5):
        sns.heatmap(out[j+5*i,:,:,0], vmax = .8, square = True, ax = ax[i, j])
        ax[i, j].set(xlabel = '', ylabel = '', title = '');
    fig.suptitle('Output after batch %d' % network_batch)
    plt.savefig(filename)
    plt.close("all")

  def get_batch(self, origin = 'train', size = 32):
    if origin == 'train':
      x = self.x_train
    else:
      x = self.x_test
    np.random.shuffle(x)
    x_batch = x[0:size,:,:, np.newaxis]
    return x_batch

  def train(self, prefix, result_dir, network_dir):
    # algorithm:
    # 0) Train adv. to guess fake vs real (freezing gen.)
    # 1) Train gen. to fool adv. (freezing adv.)
    self.critic_gp_loss_train = np.array([])
    self.critic_loss_train = np.array([])
    self.critic_loss_fake_train = np.array([])
    self.critic_loss_real_train = np.array([])
    positive_y = np.ones(self.n_batch)
    for epoch in range(self.n_iteration):
      # step critic
      n_critic = self.n_critic
      for k in range(0, n_critic):
        x_batch = self.get_batch(size = self.n_batch)
        z_batch = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_batch, self.n_pix, self.n_dimensions))

        self.generator.trainable = False
        self.critic.trainable = True
        self.gen_fixed_critic.train_on_batch([x_batch, z_batch, positive_y],
                                              [positive_y, positive_y],
                                              sample_weight = [positive_y, positive_y])

      # step generator
      z_batch = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_batch, self.n_pix, self.n_dimensions))

      self.generator.trainable = True
      self.critic.trainable = False
      self.gen_critic_fixed.train_on_batch([z_batch],
                                            [positive_y],
                                            sample_weight = [positive_y])
  
      if epoch % self.n_eval == 0:
        critic_metric_fake = 0
        critic_metric_real = 0
        c = 0.0
        for k in range(32):
          z = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_batch, self.n_pix, self.n_dimensions))
          critic_metric_fake += np.mean(self.critic.predict(self.generator.predict(z, verbose = 0), verbose = 0))
          c += 1.0
        critic_metric_fake /= c
        c = 0.0
        for k in range(32):
          x = self.get_batch(origin = 'test', size = self.n_batch)
          critic_metric_real += np.mean(self.critic.predict(x))
          c += 1.0
        critic_metric_real /= c
        critic_metric = critic_metric_fake - critic_metric_real
        if critic_metric == 0: critic_metric = 1e-20

        critic_gradient_penalty = 0
        for k in range(0, self.n_critic):
          x_batch = self.get_batch(origin = 'test', size = self.n_batch)
          z_batch = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_batch, self.n_pix, self.n_dimensions))

          self.generator.trainable = False
          self.critic.trainable = True
          critic_gradient_penalty += self.gen_fixed_critic.evaluate([x_batch, z_batch, positive_y],
                                                                    [positive_y, positive_y],
                                                                    sample_weight = [positive_y, positive_y], verbose = 0)[-1]
        critic_gradient_penalty /= float(self.n_critic)
        if critic_gradient_penalty == 0: critic_gradient_penalty = 1e-20

        self.critic_loss_train = np.append(self.critic_loss_train, [critic_metric])
        self.critic_loss_fake_train = np.append(self.critic_loss_fake_train, [critic_metric_fake])
        self.critic_loss_real_train = np.append(self.critic_loss_real_train, [critic_metric_real])
        self.critic_gp_loss_train = np.append(self.critic_gp_loss_train, [critic_gradient_penalty])
        floss = h5py.File('%s/%s_loss.h5' % (result_dir, prefix), 'w')
        floss.create_dataset('critic_loss', data = self.critic_loss_train)
        floss.create_dataset('critic_loss_real', data = self.critic_loss_real_train)
        floss.create_dataset('critic_loss_fake', data = self.critic_loss_fake_train)
        floss.create_dataset('critic_gp_loss', data = self.critic_gp_loss_train)
        floss.close()

        print("Batch %5d: L_{critic} = %10.7f ; L_{critic,fake} = %10.7f ; L_{critic,real} = %10.7f ; lambda_{gp} (|grad C| - 1)^2 = %10.7f" % (epoch, critic_metric, critic_metric_fake, critic_metric_real, self.lambda_gp*critic_gradient_penalty))
        self.save("%s/%s_generator_%d" % (network_dir, prefix, epoch), "%s/%s_critic_%d" % (network_dir, prefix, epoch))
      #gc.collect()

    print("============ End of training ===============")

  def load_loss(self, filename):
    floss = h5py.File(filename)
    self.critic_loss_train = floss['critic_loss'][:]
    self.critic_loss_real_train = floss['critic_loss_real'][:]
    self.critic_loss_fake_train = floss['critic_loss_fake'][:]
    self.critic_gp_loss_train = floss['critic_gp_loss'][:]
    self.n_iteration = self.n_eval*len(self.critic_loss_train)
    floss.close()

  def plot_train_metrics(self, filename, nnTaken = -1, epochs = True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    it = np.arange(0, self.n_iteration, self.n_eval, dtype = np.float32)
    s = 1.0
    if epochs:
      Ntrain = self.x_train.shape[0]
      s = float(self.n_batch)/float(Ntrain)
    it *= s
    plt.plot(it, smoothen(np.fabs(-self.critic_loss_train)), color = 'b', label = r' | $\mathcal{L}_{\mathrm{critic}} |$')
    plt.plot(it, smoothen(self.lambda_gp*self.critic_gp_loss_train), color = 'grey', label = r'$\lambda_{\mathrm{gp}} (||\nabla_{\hat{x}} C(\hat{x})||_{2} - 1)^2$')
    if nnTaken > 0:
      plt.axvline(x = nnTaken*s, color = 'r', linestyle = '--', label = 'Configuration taken for further analysis')
    if epochs:
      ax.set(xlabel='Epoch', ylabel='Loss', title='Training evolution');
    else:
      ax.set(xlabel='Batches', ylabel='Loss', title='Training evolution');
    ax.set_ylim([1e-1, 10])
    ax.set_yscale('log')
    plt.legend(frameon = False)
    plt.savefig(filename)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    fac = 1.0
    if np.max(np.abs(self.critic_loss_fake_train)) > np.max(np.abs(self.critic_loss_real_train)):
      fac /= np.max(np.abs(self.critic_loss_fake_train))
    else:
      fac /= np.max(np.abs(self.critic_loss_real_train))
    plt.plot(it, smoothen(fac*self.critic_loss_fake_train), color = 'g', label = r' $ %4.2f \mathcal{L}_{\mathrm{critic,fake}}$' % (fac) )
    plt.plot(it, smoothen(-fac*self.critic_loss_real_train), color = 'c', label = r' $ %4.2f \mathcal{L}_{\mathrm{critic,real}}$' % (-fac) )
    if nnTaken > 0:
      plt.axvline(x = nnTaken, color = 'r', linestyle = '--', label = 'Configuration taken for further analysis')
    ax.set(xlabel='Batches', ylabel='Loss', title='Training evolution');
    ax.set_ylim([-1, 1])
    plt.legend(frameon = False)
    filename_crit = filename.replace('.pdf', '_critic_split.pdf')
    plt.savefig(filename_crit)
    plt.close(fig)
  
  def save(self, generator_filename, critic_filename):
    critic_json = self.critic.to_json()
    with open("%s.json" % critic_filename, "w") as json_file:
      json_file.write(critic_json)
    self.critic.save_weights("%s.h5" % critic_filename)

    generator_json = self.generator.to_json()
    with open("%s.json" % generator_filename, "w") as json_file:
      json_file.write(generator_json)
    self.generator.save_weights("%s.h5" % generator_filename)

  '''
  Load stored network
  '''
  def load_generator(self, generator_filename):
    json_file = open('%s.json' % generator_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.generator = K.models.model_from_json(loaded_model_json, custom_objects={'LayerNormalization': LayerNormalization, 'GenerateImage': GenerateImage})
    self.generator.load_weights("%s.h5" % generator_filename)
    #self.generator.compile(loss = K.losses.mean_squared_error, optimizer = K.optimizers.Adam(lr = 1e-4), metrics = [])

  '''
  Load stored network
  '''
  def load(self, generator_filename, critic_filename):
    json_file = open('%s.json' % generator_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.generator = K.models.model_from_json(loaded_model_json, custom_objects={'LayerNormalization': LayerNormalization, 'GenerateImage': GenerateImage})
    self.generator.load_weights("%s.h5" % generator_filename)

    json_file = open('%s.json' % critic_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.critic = K.models.model_from_json(loaded_model_json, custom_objects = {'LayerNormalization': LayerNormalization})
    self.critic.load_weights("%s.h5" % critic_filename)

    self.critic_input = K.layers.Input(shape = (self.n_x, self.n_y), name = 'critic_input')
    self.generator_input = K.layers.Input(shape = (None, self.n_dimensions,), name = 'generator_input')

    self.generator.compile(loss = K.losses.mean_squared_error, optimizer = K.optimizers.Adam(lr = 1e-4), metrics = [])
    self.critic.compile(loss = wasserstein_loss,
                        optimizer = K.optimizers.Adam(lr = 1e-4), metrics = [])
    self.create_networks()

def main():
  import argparse

  parser = argparse.ArgumentParser(description = 'Train a Wasserstein GAN with gradient penalty to generate MNIST signal.')
  parser.add_argument('--network-dir', dest='network_dir', action='store',
                    default='network',
                    help='Directory where networks are saved during training. (default: "network")')
  parser.add_argument('--result-dir', dest='result_dir', action='store',
                    default='result',
                    help='Directory where results are saved. (default: "result")')
  parser.add_argument('--load-trained', dest='trained', action='store',
                    default='5000',
                    help='Number to be appended to end of filename when loading pretrained networks. Ignored during the "train" mode. (default: "1500")')
  parser.add_argument('--prefix', dest='prefix', action='store',
                    default='rnnwgangp',
                    help='Prefix to be added to filenames when producing plots. (default: "rnnwgangp")')
  parser.add_argument('--mode', metavar='MODE', choices=['train', 'plot_loss', 'plot_gen', 'plot_data'],
                     default = 'train',
                     help='The mode is either "train" (a neural network), "plot_loss" (plot the loss function of a previous training), "plot_gen" (show samples from the generator), "plot_data" (plot examples of the training data sample). (default: train)')
  args = parser.parse_args()
  prefix = args.prefix
  trained = args.trained

  network = RNNWGANGP()

  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
  if not os.path.exists(args.network_dir):
    os.makedirs(args.network_dir)

  # read it from disk
  network.read_input_from_files()

  if args.mode == 'plot_data':
    network.plot_data("%s/%s_data.pdf" % (args.result_dir, prefix))
  elif args.mode == 'train': # when training make some debug plots and prepare the network
    # create network
    network.create_networks()

    # for comparison: make a plot of the NN output value before any training
    # this will just be random!
    network.plot_generator_output("%s/%s_generator_output_before_training.pdf" % (args.result_dir, prefix))

    # train it
    print("Training.")
    network.train(prefix, args.result_dir, args.network_dir)

    # plot training evolution
    print("Plotting train metrics.")
    network.plot_train_metrics("%s/%s_training.pdf" % (args.result_dir, prefix))

    network.plot_generator_output("%s/%s_generator_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_loss':
    network.load_loss("%s/%s_loss.h5" % (args.result_dir, prefix))
    network.plot_train_metrics("%s/%s_training.pdf" % (args.result_dir, prefix), int(trained))
  elif args.mode == 'plot_gen':
    print("Loading network.")
    network.load_generator("%s/%s_generator_%s" % (args.network_dir, prefix, trained))
    network.plot_generator_output("%s/%s_generator_output.pdf" % (args.result_dir, prefix), int(trained))
    from shutil import copyfile
    for suf in ['h5', 'json']:
      copyfile("%s/%s_generator_%s.%s" % (args.network_dir, prefix, trained, suf), "%s/%s_generator.%s" % (args.result_dir, prefix, suf))
      copyfile("%s/%s_critic_%s.%s" % (args.network_dir, prefix, trained, suf), "%s/%s_critic.%s" % (args.result_dir, prefix, suf))
  elif args.mode == 'plot_data':
    network.plot_data("%s/%s_data.pdf" % (args.result_dir, prefix))
  else:
    print("I cannot understand the mode ", args.mode)

if __name__ == '__main__':
  main()
