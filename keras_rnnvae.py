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

def kldiv_loss(y_true, y_pred, z_mean, z_logsigma2):
  z_sigma2 = tf.exp(z_logsigma2)
  return tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) + z_sigma2 - z_logsigma2 - 1., axis = 1), axis = 0)

def kldiv_loss_mean(y_true, y_pred):
  z_mean = y_pred
  return tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(z_mean) - 1., axis = 1), axis = 0)

def kldiv_loss_stddev(y_true, y_pred):
  z_logsigma2 = y_pred
  z_sigma2 = tf.exp(z_logsigma2)
  return tf.reduce_mean(0.5 * tf.reduce_sum(z_sigma2 - z_logsigma2, axis = 1), axis = 0)

def rec_loss(y_true, y_pred):
  n_x = tf.shape(y_true)[1]
  n_y = tf.shape(y_true)[2]
  n_l = tf.shape(y_true)[3]
  N = n_x*n_y*n_l
  y_true_r = tf.reshape(y_true, [-1, N])
  y_pred_r = tf.reshape(y_pred, [-1, N])
  return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred_r - y_true_r), axis = 1), axis = 0)
  #return tf.cast(N, tf.float32)*K.losses.binary_crossentropy(y_true_r, y_pred_r)

class GenerateSamples(K.layers.Layer):

    def __init__(self,
                 **kwargs):
        """Generate Gaussian samples from a mean and exp(sigma).
        :param kwargs:
        """
        super(GenerateSamples, self).__init__(**kwargs)

    def get_config(self):
        config = {
        }
        base_config = super(GenerateSamples, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])

    def build(self, input_shape):
        super(GenerateSamples, self).build(input_shape)

    def call(self, inputs, training=None):
        z_mean = inputs[0]
        z_logsigma2 = inputs[1]
        z_sigma = tf.sqrt(1e-7 + tf.exp(z_logsigma2))

        Nbatch = K.backend.shape(z_mean)[0]
        Ndim = K.backend.shape(z_mean)[1]

        normal = tf.random.normal([Nbatch, Ndim], 0., 1.)
        z = z_mean + normal*z_sigma

        return z

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

        # This is the new setup:
        # if the pos_x and pos_y layers generate a softmax output (shape = (Nbatch, Npix, nx/y))
        # Begin -----
        pos_x = inputs[0]
        pos_y = inputs[1]
        energy = inputs[2]
        # get batch size
        Nbatch = K.backend.shape(energy)[0]
        # get number of pixels
        Npix = self.n_pix # K.backend.shape(energy)[1]

        pos_x_tiled = tf.tile(tf.reshape(pos_x, [Nbatch, Npix, self.n_x, 1]), [1, 1, 1, self.n_y])
        pos_y_tiled = tf.tile(tf.reshape(pos_y, [Nbatch, Npix, 1, self.n_y]), [1, 1, self.n_x, 1])
        pos_rows = pos_x_tiled*pos_y_tiled

        # now pos_rows has shape (Nbatch, Npix, nx, ny)
        # we can scale it by energy to have one image with the correct energy per pixel (blurred out depending on how certain the network is to output a single pixel or many)
        energy_tiled = tf.tile(tf.tile(tf.reshape(energy, [Nbatch, Npix, 1, 1]), [1, 1, self.n_x, 1]), [1, 1, 1, self.n_y])

        # now sum all pixels
        image = tf.reduce_sum(pos_rows * energy_tiled, axis = 1) # sums over pixels, as the shape of both is (Nbatch, Npix, nx, ny)
        image = tf.reshape(image, [Nbatch, self.n_x, self.n_y, 1]) # add axis for layer

        # End -----

        return image

class RNNVAE(object):
  '''
  Implementation of the variational auto-encoder.

  The following procedure is implemented:
  1) Minimize the reconstruction loss and the KL-loss simultaneously
  '''

  def __init__(self, n_iteration = 20000,
               n_batch = 128,
               n_eval = 50,
               n_x = 28, n_y = 28,
               n_dimensions = 256):
    '''
    Initialise the network.

    :param n_iteration: Number of batches to run over in total.
    :param n_batch: Number of samples in a batch.
    :param n_eval: Number of batches to train before evaluating metrics.
    '''
    self.n_iteration = n_iteration
    self.n_batch = n_batch
    self.n_eval = n_eval
    self.enc = None
    self.dec = None
    self.n_x = n_x
    self.n_y = n_y
    self.n_dimensions = n_dimensions

  '''
    Create encoder network.
  '''
  def create_enc(self):
    self.enc_input = Input(shape = (self.n_x, self.n_y, 1), name = 'enc_input')

    xc = self.enc_input

    xc = K.layers.Conv2D(32, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.MaxPooling2D(pool_size = (2, 2))(xc)

    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.MaxPooling2D(pool_size = (2, 2))(xc)

    xc = K.layers.Flatten()(xc)

    xc = K.layers.Dense(512, activation = None, name = "adv_6")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    #xc = K.layers.Dropout(0.5)(xc)
    xc = K.layers.Dense(128, activation = None, name = "adv_7")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Dense(64, activation = None, name = "adv_8")(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)

    z_mean = K.layers.Dense(self.n_dimensions, activation = None)(xc)
    z_logsigma2 = K.layers.Dense(self.n_dimensions, activation = None)(xc)

    self.enc = Model(self.enc_input, [z_mean, z_logsigma2], name = "enc")
    self.enc.trainable = True
    self.enc.compile(loss = [K.losses.mean_squared_error, K.losses.mean_squared_error],
                     optimizer = Adam(lr = 1e-3), metrics = [])

  '''
  Create decoder network.
  '''
  def create_dec(self):
    self.dec_input = Input(shape = (self.n_dimensions,), name = 'dec_input')

    xg = self.dec_input

    xg = K.layers.Reshape((self.n_dimensions, 1))(xg)

    xg = K.layers.recurrent.LSTM(256, return_sequences = True)(xg)
    xg = K.layers.recurrent.LSTM(128, return_sequences = True)(xg)
    pos_x = K.layers.TimeDistributed(K.layers.Dense(self.n_x, activation = 'softmax'))(xg)
    pos_y = K.layers.TimeDistributed(K.layers.Dense(self.n_y, activation = 'softmax'))(xg)
    energy = K.layers.TimeDistributed(K.layers.Dense(1, activation = 'relu'))(xg)
    xg = GenerateImage(self.n_x, self.n_y, self.n_dimensions)([pos_x, pos_y, energy])

    xg = K.layers.Conv2DTranspose(32, (3,3), padding = "same", activation = None)(xg)
    xg = K.layers.LeakyReLU(0.2)(xg)
    xg = K.layers.Conv2DTranspose(32, (3,3), padding = "same", activation = None)(xg)
    xg = K.layers.LeakyReLU(0.2)(xg)

    xg = K.layers.Conv2DTranspose(16, (3,3), padding = "same", activation = None)(xg)
    xg = K.layers.LeakyReLU(0.2)(xg)
    xg = K.layers.Conv2DTranspose(16, (3,3), padding = "same", activation = None)(xg)
    xg = K.layers.LeakyReLU(0.2)(xg)
    #xg = K.layers.Dropout(0.5)(xg)

    xg = K.layers.Conv2DTranspose(8, (3,3), padding = "same", activation = None)(xg)
    xg = K.layers.LeakyReLU(0.2)(xg)
    xg = K.layers.Conv2DTranspose(1, (3,3), padding = "same", activation = None)(xg)
    xg = K.layers.ReLU()(xg)
    #xg = K.layers.Dropout(0.5)(xg)
    #xg = K.layers.Activation('sigmoid')(xg)

    self.dec = Model(self.dec_input, xg, name = "dec")
    self.dec.trainable = True
    self.dec.compile(loss = K.losses.mean_squared_error, optimizer = Adam(lr = 1e-4), metrics = [])

  '''
  Create all networks.
  '''
  def create_networks(self):
    if not self.enc:
      self.create_enc()
    if not self.dec:
      self.create_dec()

    self.dec.trainable = True
    self.enc.trainable = True

    self.real_input = Input(shape = (self.n_x, self.n_y, 1), name = 'real_input')

    self.z_mean, self.z_logsigma2 = self.enc(self.real_input)
    self.z_generated = GenerateSamples()([self.z_mean, self.z_logsigma2])

    from functools import partial
    kldiv_loss_partial = partial(kldiv_loss, z_mean = self.z_mean, z_logsigma2 = self.z_logsigma2)

    self.vae = Model(self.real_input,
                     [self.dec(self.z_generated), self.z_mean, self.z_logsigma2],
                     name = "vae")
    self.vae.compile(loss = [rec_loss, kldiv_loss_mean, kldiv_loss_stddev],
                     loss_weights = [1.0, 1.0, 1.0],
                     optimizer = Adam(lr = 1e-4), metrics = [])

    print("Encoder:")
    self.enc.trainable = True
    self.enc.summary()
    print("Decoder:")
    self.dec.trainable = True
    self.dec.summary()
    print("VAE:")
    self.vae.summary()

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

  def plot_decoder_output(self, filename, network_batch = 0):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize = (20, 20), nrows = 5, ncols = 5)
    z = np.random.normal(loc = 0.0, scale = 1.0, size = (5*5, self.n_dimensions,))
    out = self.dec.predict(z, verbose = 0)
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
    self.rec_loss_train = np.array([])
    self.kl_loss_train = np.array([])
    positive_y = np.ones(self.n_batch)
    for epoch in range(self.n_iteration):
      x_batch = self.get_batch(size = self.n_batch)
      self.vae.train_on_batch([x_batch],
                              [x_batch, positive_y, positive_y])
  
      if epoch % self.n_eval == 0:
        rec_loss = 0
        kl_loss = 0
        x_batch = self.get_batch(origin = 'test', size = self.n_batch)

        vae_loss, rec_loss, kl_loss_mean, kl_loss_stddev = self.vae.evaluate([x_batch],
                                                                             [x_batch, positive_y, positive_y],
                                                                             verbose = 0)
        kl_loss = kl_loss_mean + kl_loss_stddev
        self.rec_loss_train = np.append(self.rec_loss_train, [rec_loss])
        self.kl_loss_train = np.append(self.kl_loss_train, [kl_loss])
        floss = h5py.File('%s/%s_loss.h5' % (result_dir, prefix), 'w')
        floss.create_dataset('rec_loss', data = self.rec_loss_train)
        floss.create_dataset('kl_loss', data = self.kl_loss_train)
        floss.close()

        print("Batch %5d: L_{VAE} = %20.16f ; L_{rec} = %20.16f ; L_{KL} = %20.16f" % (epoch, vae_loss, rec_loss, kl_loss))
        self.save("%s/%s_enc_%d" % (network_dir, prefix, epoch), "%s/%s_dec_%d" % (network_dir, prefix, epoch))
      #gc.collect()

    print("============ End of training ===============")

  def load_loss(self, filename):
    floss = h5py.File(filename)
    self.rec_loss_train = floss['rec_loss'][:]
    self.kl_loss_train = floss['kl_loss'][:]
    self.n_iteration = self.n_eval*len(self.rec_loss_train)
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
    plt.plot(it, smoothen(np.fabs(self.rec_loss_train)), color = 'b', label = r' | $\mathcal{L}_{\mathrm{rec}}$ |')
    plt.plot(it, smoothen(np.fabs(self.kl_loss_train)), color = 'grey', label = r'| $\mathcal{L}_{\mathrm{KL}}$ |' )
    plt.plot(it, smoothen(np.fabs(self.rec_loss_train + self.kl_loss_train)), color = 'k', label = r'| $\mathcal{L}_{\mathrm{VAE}}$ |' )
    if nnTaken > 0:
      plt.axvline(x = nnTaken*s, color = 'r', linestyle = '--', label = 'Configuration taken for further analysis')
    if epochs:
      ax.set(xlabel='Epoch', ylabel='Loss', title='Training evolution');
    else:
      ax.set(xlabel='Batches', ylabel='Loss', title='Training evolution');
    ax.set_ylim([1e-1, 100])
    ax.set_yscale('log')
    plt.legend(frameon = False)
    plt.savefig(filename)
    plt.close(fig)
  
  def save(self, enc_filename, dec_filename):
    enc_json = self.enc.to_json()
    with open("%s.json" % enc_filename, "w") as json_file:
      json_file.write(enc_json)
    self.enc.save_weights("%s.h5" % enc_filename)

    dec_json = self.dec.to_json()
    with open("%s.json" % dec_filename, "w") as json_file:
      json_file.write(dec_json)
    self.dec.save_weights("%s.h5" % dec_filename)

  '''
  Load stored network
  '''
  def load_decoder(self, dec_filename):
    json_file = open('%s.json' % dec_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.dec = K.models.model_from_json(loaded_model_json, custom_objects={'LayerNormalization': LayerNormalization})
    self.dec.load_weights("%s.h5" % dec_filename)
    #self.dec.compile(loss = K.losses.mean_squared_error, optimizer = K.optimizers.Adam(lr = 1e-4), metrics = [])

  '''
  Load stored network
  '''
  def load(self, enc_filename, dec_filename):
    json_file = open('%s.json' % enc_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.enc = K.models.model_from_json(loaded_model_json, custom_objects={'LayerNormalization': LayerNormalization, 'GenerateSamples': GenerateSamples})
    self.enc.load_weights("%s.h5" % enc_filename)

    json_file = open('%s.json' % dec_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.dec = K.models.model_from_json(loaded_model_json, custom_objects = {'LayerNormalization': LayerNormalization})
    self.dec.load_weights("%s.h5" % dec_filename)

    self.enc_input = Input(shape = (self.n_x, self.n_y, 1), name = 'enc_input')
    self.dec_input = Input(shape = (self.n_dimensions,), name = 'dec_input')

    self.enc.compile(loss = [K.losses.mean_squared_error, K.losses.mean_squared_error], optimizer = K.optimizers.Adam(lr = 1e-4), metrics = [])
    self.dec.compile(loss = K.losses.mean_squared_error, optimizer = K.optimizers.Adam(lr = 1e-4), metrics = [])
    self.create_networks()

def main():
  import argparse

  parser = argparse.ArgumentParser(description = 'Train a VAE to generate MNIST signal.')
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
                    default='rnnvae',
                    help='Prefix to be added to filenames when producing plots. (default: "rnnvae")')
  parser.add_argument('--mode', metavar='MODE', choices=['train', 'plot_loss', 'plot_gen', 'plot_data'],
                     default = 'train',
                     help='The mode is either "train" (a neural network), "plot_loss" (plot the loss function of a previous training), "plot_gen" (show samples from the decoder), "plot_data" (plot examples of the training data sample). (default: train)')
  args = parser.parse_args()
  prefix = args.prefix
  trained = args.trained

  network = RNNVAE()

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
    network.plot_decoder_output("%s/%s_decoder_output_before_training.pdf" % (args.result_dir, prefix))

    # train it
    print("Training.")
    network.train(prefix, args.result_dir, args.network_dir)

    # plot training evolution
    print("Plotting train metrics.")
    network.plot_train_metrics("%s/%s_training.pdf" % (args.result_dir, prefix))

    network.plot_decoder_output("%s/%s_decoder_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_loss':
    network.load_loss("%s/%s_loss.h5" % (args.result_dir, prefix))
    network.plot_train_metrics("%s/%s_training.pdf" % (args.result_dir, prefix), int(trained))
  elif args.mode == 'plot_gen':
    print("Loading network.")
    network.load_decoder("%s/%s_dec_%s" % (args.network_dir, prefix, trained))
    network.plot_decoder_output("%s/%s_decoder_output.pdf" % (args.result_dir, prefix), int(trained))
    from shutil import copyfile
    for suf in ['h5', 'json']:
      copyfile("%s/%s_dec_%s.%s" % (args.network_dir, prefix, trained, suf), "%s/%s_dec.%s" % (args.result_dir, prefix, suf))
      copyfile("%s/%s_enc_%s.%s" % (args.network_dir, prefix, trained, suf), "%s/%s_enc.%s" % (args.result_dir, prefix, suf))
  elif args.mode == 'plot_data':
    network.plot_data("%s/%s_data.pdf" % (args.result_dir, prefix))
  else:
    print("I cannot understand the mode ", args.mode)

if __name__ == '__main__':
  main()
