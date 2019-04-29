
import tensorflow as tf
import keras as K

'''
  Given an image of the jet at shower time t, calculate:
  z = ratio between highest energy pixel and the total energy in the image
  Pq_qg  = Cf*(1+z**2)/(1-z)           # prob. of q -> q g
  Pg_gg  = Nc*(1-z*(1-z))**2/(z*(1-z))  # prob. of g -> g g
  Pg_qqb = Tr*(z**2 + (1-z)**2)         # prob. of g -> q + qbar
  Pq_qp  = eq**2 * (1+z**2)/(1-z)       # prob. of q -> q + photon

  with:
  Cf = 4/3
  Nc = 3
  Tr = nf/2 = 5/2 (assuming 5 flavours)
  eq = charge of quark -> will be set to 1 here, as we are only interested in the correlations
                          and there is no way of knowing if the splitting came from an up or down flavour

  Assumes that the image has a shape (Nbatch, Npix, nx*ny*1).
  The outputs will have shapes (Nbatch, Npix, 1).
'''
def get_splitting_function_values_tf(image):
  Cf = 4.0/3.0
  Nc = 3.0
  Tr = 5.0/2.0
  Nbatch = tf.shape(image)[0]
  Npix = tf.shape(image)[1]

  Etot = tf.reduce_sum(image, axis = 2) # shape = (Nbatch, Npix)
  # example, ignoring batch axis: Etot = [1, 2, 3]
  # Etot_cum = [1+2+3, 2+3, 3]
  # i-th element contains sum of all elements with indices >= i
  Etot_cum = tf.cumsum(Etot_per_image, axis = 1, reverse = True) + 1e-6 # shape = (Nbatch, Npix)
  
  z = tf.reshape(Etot/Etot_cum, [Nbatch, Npix, 1])
  
  Pq_qg  = Cf*(1+z**2)/(1-z)
  Pg_gg  = Nc*(1-z*(1-z))**2/(z*(1-z))
  Pg_qqb = Tr*(z**2 + (1-z)**2)
  Pq_qp  = (1+z**2)/(1-z)

  return [z, Pq_qg, Pg_gg, Pg_qqb, Pq_qp]


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
        # This is the new setup:
        # pos_xy generate a softmax output (shape = (Nbatch, Npix, nx*ny))
        # Begin -----
        pos_xy = inputs[0]
        energy = inputs[1]
        # get batch size
        Nbatch = K.backend.shape(energy)[0]
        # get number of pixels
        Npix = self.n_pix # K.backend.shape(energy)[1]

        energy_tiled = tf.tile(energy, [1, 1, self.n_x*self.n_y])

        image_per_time = pos_xy * energy_tiled
        # get t-dependent splitting function-related quantities
        # all of them have shape (Nbatch, Npix, 1)
        z, Pq_qg, Pg_gg, Pg_qqb, Pq_qp = get_splitting_function_values_tf(image_per_time)

        # now sum all pixels
        image = tf.reduce_sum(image_per_time, axis = 1) # sums over pixels, as the shape of both is (Nbatch, Npix, nx*ny)
        image = tf.reshape(image, [Nbatch, self.n_x, self.n_y, 1]) # add axis for layer
        # image has shape (Nbatch, nx, ny, 1)

        # End -----

        return image, z, Pq_qg, Pg_gg, Pg_qqb, Pq_qp
