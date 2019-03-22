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

mnist = K.datasets.mnist

def smoothen(y):
  N = 20
  box = np.ones(N)/float(N)
  return np.convolve(y, box, mode = 'same')

def gradient_penalty_loss(y_true, y_pred, critic, generator, zc, z, real, N, n_x, n_y):
  d1 = generator([z, zc])
  d2 = real
  diff = d2 - d1
  epsilon = K.backend.random_uniform_variable(shape=[N, n_x, n_y, 1], low = 0., high = 1.)
  interp_input = d1 + (epsilon*diff)
  gradients = K.backend.gradients(critic(interp_input), [interp_input])[0]
  ## not needed as there is a single element in interp_input here (the discriminator output)
  ## the only dimension left is the batch, which we just average over in the last step
  slopes = K.backend.sqrt(1e-6 + K.backend.sum(K.backend.square(gradients), axis = [1]))
  gp = K.backend.mean(K.backend.square(1 - slopes))
  return gp

def wasserstein_loss(y_true, y_pred):
  return K.backend.mean(y_true*y_pred, axis = 0)

def log_loss(y_true, y_pred):
  return -K.backend.mean(y_true*K.backend.log(1e-8 + y_pred), axis = 0)

class DMWGANGP(object):
  '''
  Implementation of the Wasserstein GAN with Gradient Penalty algorithm to generate samples of a training set.
  Ref.: https://github.com/mahyarkoy/dmgan_release/blob/master/tf_dmgan.py
  Ref.: https://arxiv.org/abs/1806.00880
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
               n_batch = 32,
               lambda_gp = 10.0,
               lambda_enc = 1.0,
               n_eval = 10,
               n_x = 28, n_y = 28,
               n_dimensions = 200,
               n_gens = 10):
    '''
    Initialise the network.

    :param n_iteration: Number of batches to run over in total.
    :param n_critic: Number of batches to train the critic on per batch of the discriminator.
    :param n_batch: Number of samples in a batch.
    :param lambda_gp: Lambda parameter to weight the gradient penalty of the critic loss function.
    :param lambda_enc: Lambda parameter to weight the encoder entropy.
    :param n_eval: Number of batches to train before evaluating metrics.
    :param n_dimensions: Dimension of latent space.
    :param n_gens: Number of generators to produce.
    '''
    self.n_iteration = n_iteration
    self.n_critic = n_critic
    self.n_batch = n_batch
    self.lambda_gp = lambda_gp
    self.lambda_enc = lambda_enc
    self.n_eval = n_eval
    self.critic = {}
    self.generator = {}
    self.encoder = {} # network to parametrise the pdf for the generator selection
    self.n_x = n_x
    self.n_y = n_y
    self.n_dimensions = n_dimensions
    self.n_gens = n_gens


  '''
    Create inputs.
  '''
  def create_inputs(self):
    # dummy inputs used to create model blocks
    self.generator_input = Input(shape = (self.n_dimensions,), name = 'generator_input') # latent space variable used for generator input
    self.critic_input = Input(shape = (self.n_x, self.n_y, 1), name = 'critic_input')    # image input for critic
    self.encoder_input = Input(shape = (self.n_x, self.n_y, 1), name = 'encoder_input')  # input for encoder

    # inputs used in the generator-critic combined systems
    self.dummy_input = Input(shape = (1,), name = 'dummy_input')                         # dummy unused input to add gradient penalty
    self.z_input = Input(shape = (self.n_dimensions,), name = 'z_input')                 # latent space
    self.real_input = Input(shape = (self.n_x, self.n_y, 1), name = 'real_input')        # image input

    # input used to select generator
    self.zc_input = Input(shape = (1,), name = 'zc_input', dtype = 'int32')              # latent variable to select generator

  '''
    Create a new critic network.
  '''
  def create_critic(self, n = None):
    if n == None:
      n = len(self.critic)

    xc = self.critic_input

    xc = K.layers.Conv2D(32, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.MaxPooling2D(pool_size = (2, 2), name = "adv_2")(xc)
    #xc = K.layers.Dropout(0.5)(xc)

    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.MaxPooling2D(pool_size = (2, 2))(xc)
    #xc = K.layers.Dropout(0.5)(xc)

    xc = K.layers.Flatten()(xc)
    xc = K.layers.Dense(512, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    #xc = K.layers.Dropout(0.5)(xc)
    xc = K.layers.Dense(128, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Dense(64, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Dense(1, activation = None)(xc)

    self.critic[n] = Model(self.critic_input, xc, name = "c_%d" % n)
    self.critic[n].trainable = True
    self.critic[n].compile(loss = wasserstein_loss,
                           optimizer = Adam(lr = 1e-4), metrics = [])

  '''
  Create a new generator network.
  '''
  def create_generator(self, n = None):
    if n == None:
      n = len(self.generator)

    xg = self.generator_input

    xg = K.layers.Dense(512, activation = None)(xg)
    xg = K.layers.LeakyReLU(0.2)(xg)
    xg = K.layers.Dense(256, activation = None)(xg)
    xg = K.layers.LeakyReLU(0.2)(xg)
    xg = K.layers.Dense(128, activation = None)(xg)
    xg = K.layers.LeakyReLU(0.2)(xg)
    xg = K.layers.Dense(self.n_x*self.n_y*1, activation = None)(xg)
    xg = K.layers.LeakyReLU(0.2)(xg)
    #xg = K.layers.Dropout(0.5)(xg)

    xg = K.layers.Reshape((self.n_x, self.n_y, 1))(xg)

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
    xg = K.layers.LeakyReLU(0.2)(xg)

    self.generator[n] = Model(self.generator_input, xg, name = "g_%d" % n)
    self.generator[n].trainable = True
    self.generator[n].compile(loss = K.losses.mean_squared_error,
                              optimizer = Adam(lr = 1e-4), metrics = [])

  def create_encoder(self, n = None):
    if n == None:
      n = len(self.encoder)

    xc = self.encoder_input

    xc = K.layers.Conv2D(32, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.MaxPooling2D(pool_size = (2, 2), name = "enc_2")(xc)

    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Conv2D(16, (3,3), padding = "same", activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.MaxPooling2D(pool_size = (2, 2))(xc)

    xc = K.layers.Flatten()(xc)

    xc = K.layers.Dense(512, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Dense(128, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Dense(64, activation = None)(xc)
    xc = K.layers.LeakyReLU(0.2)(xc)
    xc = K.layers.Dense(1, activation = None)(xc)

    self.encoder[n] = Model(self.encoder_input, xc, name = "e_%d" % n)
    self.encoder[n].trainable = True
    self.encoder[n].compile(loss = binary_crossentropy,
                            optimizer = Adam(lr = 1e-4), metrics = [])

  '''
  Create all networks.
  '''
  def create_networks(self):
    from functools import partial

    # create inputs first
    self.create_inputs()

    # create critic if not loaded already
    if len(self.critic) == 0:
      self.create_critic(0)

    # create encoder if not loaded already
    if len(self.encoder) == 0:
      self.create_encoder(0)

    print("Critic:")
    self.critic[0].trainable = True
    self.critic[0].summary()

    # create generator if not loaded already
    if len(self.generator) == 0:
      for i in range(self.n_gens):
        self.create_generator(i)
        print("Generator %d:" % i)
        self.generator[i].trainable = True
        self.generator[i].summary()

    # combine all generators scaled by a one hot encoded tensor to select the active one
    # the combined generator, which just outputs the result of the selected generator, as given by self.zc_input,
    # is stored in self.combined_generator
    for i in range(self.n_gens):
      self.generator[i].trainable = True

    # create a list of the generator outputs for an input z_input
    self.list_generator = list()
    # put all generator outputs in a list for z_input
    for i in range(self.n_gens):
      self.list_generator.append(self.generator[i](self.z_input)) # create list of generator outputs (this is a Tensor and not a Model)

    def combine_gens(x_in):
      x = list(x_in)
      zc_input = K.backend.cast(x[0], dtype='int32')
      list_generator = x[1:]
      zc_one_hot = K.backend.one_hot(zc_input, self.n_gens)
      zc_map = K.backend.tile(zc_one_hot, [1, 1, self.n_x*self.n_y*1])
      zc_map_reshape = K.backend.reshape(zc_map, [-1, self.n_gens, self.n_x, self.n_y, 1])
      g_stack = K.backend.stack(list_generator, axis = 1)
      gen_sum = K.backend.sum(g_stack * zc_map_reshape, axis = 1)
      return gen_sum

    comb_in = [self.zc_input]
    comb_in.extend(self.list_generator)
    gen_sum_reshape = K.layers.Lambda(combine_gens)(comb_in)
    self.combined_generator = Model([self.z_input, self.zc_input],
                                    [gen_sum_reshape],
                                    name = "combined_generator")
    self.combined_generator.compile(loss = K.losses.mean_squared_error,
                                    optimizer = Adam(lr = 1e-4), metrics = [])
    print("Combined generator:")
    self.combined_generator.summary()

    # create gradient penalty for the combined generator and the critic
    for i in range(self.n_gens):
      self.generator[i].trainable = False
    self.critic[0].trainable = True
    partial_gp_loss = partial(gradient_penalty_loss,
                              critic = self.critic[0],
                              generator = self.combined_generator,
                              z = self.z_input,
                              zc = self.zc_input,
                              real = self.real_input,
                              N = self.n_batch,
                              n_x = self.n_x, n_y = self.n_y)

    # now use this combined generator to calculate the Wasserstein distance
    wdistance = K.layers.Subtract()([self.critic[0](self.combined_generator([self.z_input, self.zc_input])),
                                     self.critic[0](self.real_input)])

    # create model to train the critic
    for i in range(self.n_gens):
      self.generator[i].trainable = False
      self.combined_generator.trainable = False
    self.critic[0].trainable = True
    self.gen_fixed_critic = Model([self.real_input, self.z_input, self.zc_input, self.dummy_input],
                                  [wdistance, self.dummy_input],
                                  name = "gfc")
    self.gen_fixed_critic.compile(loss = [wasserstein_loss, partial_gp_loss],
                                  loss_weights = [1.0, self.lambda_gp],
                                  optimizer = Adam(lr = 1e-4, beta_1 = 0), metrics = [])


    # create models to train the generators
    # add a entropy term for the encoder in it:
    partial_log_loss = partial(log_loss)
    self.gen_critic_fixed = {}
    for i in range(self.n_gens):
      for k in range(self.n_gens):
        self.generator[k].trainable = False
        self.combined_generator.trainable = False
      self.critic[0].trainable = False
      self.generator[k].trainable = True
      self.gen_critic_fixed[i] = Model([self.z_input, self.zc_input],
                                       [self.critic[0](self.combined_generator([self.z_input, self.zc_input])), self.encoder[0](self.combined_generator([self.z_input, self.zc_input]))],
                                       name = "gcf_%d" % i)
      self.gen_critic_fixed[i].compile(loss = [wasserstein_loss, partial_log_loss],
                                       loss_weights = [-1.0, self.lambda_enc],
                                       optimizer = Adam(lr = 1e-4, beta_1 = 0), metrics = [])

  '''
    Read data and put it in x_train and x_test after minor preprocessing.
  '''
  def read_input_from_files(self):
    (self.x_train, self.y_train), (self.x_test, self.y_test) = K.datasets.mnist.load_data()
    self.x_train = self.x_train.astype('float32')
    self.x_test = self.x_test.astype('float32')
    self.x_train /= 255.0
    self.x_test /= 255.0

  '''
    Plot input data for comparison.
  '''
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

  '''
    Plot generator output for comparison.
  '''
  def plot_generator_output(self, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize = (20, 20), nrows = 5, ncols = 5)
    z = np.random.normal(loc = 0.0, scale = 1.0, size = (5*5, self.n_dimensions,))
    zc = np.random.randint(low = 0, high = self.n_gens, size = (5*5, 1))
    out = self.combined_generator.predict([z, zc], verbose = 0)
    for i in range(5):
      for j in range(5):
        sns.heatmap(out[j+5*i,:,:,0], vmax = .8, square = True, ax = ax[i, j])
        ax[i, j].set(xlabel = '', ylabel = '', title = '');
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
        z_batch = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_batch, self.n_dimensions))
        zc_batch = np.random.randint(low = 0, high = self.n_gens, size = (self.n_batch, 1))

        self.combined_generator.trainable = False
        self.critic[0].trainable = True
        self.gen_fixed_critic.train_on_batch([x_batch, z_batch, zc_batch, positive_y],
                                              [positive_y, positive_y],
                                              sample_weight = [positive_y, positive_y])

      # step generator
      for i in range(0, self.n_gens):
        z_batch = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_batch, self.n_dimensions))
        zc_batch = np.random.randint(low = 0, high = self.n_gens, size = (self.n_batch, 1))

        self.combined_generator.trainable = False
        for k in range(0, self.n_gens):
          self.generator[k].trainable = False
        self.generator[i].trainable = True
        self.critic[0].trainable = False
        self.gen_critic_fixed[i].train_on_batch([z_batch, zc_batch],
                                                [positive_y, positive_y],
                                                sample_weight = [positive_y, positive_y])
  
      if epoch % self.n_eval == 0:
        critic_metric_fake = 0
        critic_metric_real = 0
        c = 0.0
        for k in range(32):
          z = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_batch, self.n_dimensions))
          zc = np.random.randint(low = 0, high = self.n_gens, size = (self.n_batch, 1))
          critic_metric_fake += np.mean(self.critic.predict(self.combined_generator.predict([z, zc], verbose = 0), verbose = 0))
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
          z_batch = np.random.normal(loc = 0.0, scale = 1.0, size = (self.n_batch, self.n_dimensions))
          zc_batch = np.random.randint(low = 0, high = self.n_gens, size = (self.n_batch, 1))

          self.combined_generator.trainable = False
          self.critic.trainable = True
          critic_gradient_penalty += self.gen_fixed_critic.evaluate([x_batch, z_batch, zc_batch, positive_y],
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
        self.save("%s/%s_generator_%d" % (network_dir, prefix, epoch), "%s/%s_critic_%d" % (network_dir, prefix, epoch), "%s/%s_encoder_%d" % (network_dir, prefix, epoch))
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

  def plot_train_metrics(self, filename, nnTaken = -1):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    it = np.arange(0, self.n_iteration, self.n_eval)
    plt.plot(it, smoothen(np.fabs(-self.critic_loss_train)), color = 'b', label = r' | $\mathcal{L}_{\mathrm{critic}} |$')
    plt.plot(it, smoothen(self.lambda_gp*self.critic_gp_loss_train), color = 'grey', label = r'$\lambda_{\mathrm{gp}} (||\nabla_{\hat{x}} C(\hat{x})||_{2} - 1)^2$')
    if nnTaken > 0:
      plt.axvline(x = nnTaken, color = 'r', linestyle = '--', label = 'Configuration taken for further analysis')
    ax.set(xlabel='Batches', ylabel='Loss', title='Training evolution');
    ax.set_ylim([1e-1, 100])
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
  
  def save(self, generator_filename, critic_filename, encoder_filename):
    critic_json = self.critic[0].to_json()
    with open("%s.json" % critic_filename, "w") as json_file:
      json_file.write(critic_json)
    self.critic[0].save_weights("%s.h5" % critic_filename)

    encoder_json = self.encoder[0].to_json()
    with open("%s.json" % encoder_filename, "w") as json_file:
      json_file.write(encoder_json)
    self.encoder[0].save_weights("%s.h5" % encoder_filename)

    for i in range(self.n_gens):
      generator_json = self.generator[i].to_json()
      with open("%s_%d.json" % (generator_filename, i), "w") as json_file:
        json_file.write(generator_json)
      self.generator[i].save_weights("%s_%d.h5" % (generator_filename, i))

  '''
  Load stored network
  '''
  def load_generator(self, generator_filename):
    for i in range(self.n_gens):
      json_file = open('%s_%d.json' % (generator_filename, i), 'r')
      loaded_model_json = json_file.read()
      json_file.close()
      self.generator[i] = K.models.model_from_json(loaded_model_json, custom_objects={'LayerNormalization': LayerNormalization})
      self.generator[i].load_weights("%s_%d.h5" % (generator_filename, i))

  '''
  Load stored network
  '''
  def load(self, generator_filename, critic_filename, encoder_filename):
    for i in range(self.n_gens):
      json_file = open('%s_%d.json' % (generator_filename, i), 'r')
      loaded_model_json = json_file.read()
      json_file.close()
      self.generator[i] = K.models.model_from_json(loaded_model_json, custom_objects={'LayerNormalization': LayerNormalization})
      self.generator[i].load_weights("%s_%d.h5" % (generator_filename, i))
      self.generator[i].compile(loss = K.losses.mean_squared_error, optimizer = K.optimizers.Adam(lr = 1e-4), metrics = [])

    json_file = open('%s.json' % critic_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.critic[0] = K.models.model_from_json(loaded_model_json, custom_objects = {'LayerNormalization': LayerNormalization})
    self.critic[0].load_weights("%s.h5" % critic_filename)
    self.critic[0].compile(loss = wasserstein_loss,
                           optimizer = K.optimizers.Adam(lr = 1e-4), metrics = [])

    json_file = open('%s.json' % encoder_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.encoder[0] = K.models.model_from_json(loaded_model_json, custom_objects = {'LayerNormalization': LayerNormalization})
    self.encoder[0].load_weights("%s.h5" % encoder_filename)
    self.encoder[0].compile(loss = wasserstein_loss,
                            optimizer = K.optimizers.Adam(lr = 1e-4), metrics = [])

    self.create_networks()

def main():
  import argparse

  parser = argparse.ArgumentParser(description = 'Train a Wasserstein GAN with gradient penalty using the disconnected manifold method to generate MNIST signal.')
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
                    default='dmwgangp',
                    help='Prefix to be added to filenames when producing plots. (default: "dmwgangp")')
  parser.add_argument('--mode', metavar='MODE', choices=['train', 'plot_loss', 'plot_gen', 'plot_data'],
                     default = 'train',
                     help='The mode is either "train" (a neural network), "plot_loss" (plot the loss function of a previous training), "plot_gen" (show samples from the generator), "plot_data" (plot examples of the training data sample). (default: train)')
  args = parser.parse_args()
  prefix = args.prefix
  trained = args.trained

  network = DMWGANGP()

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
    network.load("%s/%s_generator_%s" % (args.network_dir, prefix, trained), "%s/%s_critic_%s" % (args.network_dir, prefix, trained), "%s/%s_encoder_%s" % (args.network_dir, prefix, trained))
    network.plot_generator_output("%s/%s_generator_output.pdf" % (args.result_dir, prefix))
  elif args.mode == 'plot_data':
    network.plot_data("%s/%s_data.pdf" % (args.result_dir, prefix))
  else:
    print("I cannot understand the mode ", args.mode)

if __name__ == '__main__':
  main()
