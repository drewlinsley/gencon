from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compat import v1 as tf
import tensorflow as tf2
from tf_slim import layers
from tf_slim.layers import normalization

import numpy as np

from .. import model
from .prc import feedback_hgru_dyn as feedback_hgru_v5_3l_nu_f


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - tf.mod(factor, 2)


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) / 2  # differentiable_round(size + 1, -1) / 2
    center = tf.cond(
        tf.equal(tf.mod(size, 2), 1),
        lambda: factor - 1,
        lambda: factor - 0.5,
    )
    # size = tf.cast(size, tf.int32)
    row_range, col_range = tf.reshape(tf.range(size), [-1, 1]), tf.reshape(tf.range(size), [1, -1])
    row_range = tf.cast(row_range, tf.float32)
    col_range = tf.cast(col_range, tf.float32)
    return (1 - tf.abs(row_range - center) / factor) * (1 - tf.abs(col_range - center) / factor) 


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)
    # filter_size = tf.ceil(differentiable_round(filter_size, -1))
    filter_size = differentiable_round(filter_size)
    weights = tf.ones((
        1,
        filter_size,
        filter_size,
        number_of_classes,
        number_of_classes), dtype=tf.float32)

    upsample_kernel = upsample_filt(filter_size)
    return weights * upsample_kernel[None, ..., None, None]


def upsample_tf(input_img, factor):

    # n, d, h, w, c = tf.shape(input_img)
    shape = tf.cast(tf.shape(input_img), tf.float32)
    upsample_kernel = bilinear_upsample_weights(factor=factor, number_of_classes=shape[4])

    # factor = tf.cast(differentiable_round(factor, -1), tf.int32)
    new_height = differentiable_round(shape[2] * factor)
    new_width = differentiable_round(shape[3] * factor)

    import pdb;pdb.set_trace()
    # # Alternatively, do the below in a loop
    # res = tf2.image.resize(
    #         tf.squeeze(input_img, 0),  # tf.reshape(input_img, tf.concat([[shape[0] * shape[1]], shape[2:]], 0)),
    #     [new_height, new_width],
    # )
    # res = res[None]  # tf.reshape([shape[0], shape[1], new_height, new_width, shape[4]])
    res = tf.nn.conv3d_transpose(
        input_img,
        upsample_kernel,
        padding="SAME",
        strides=[1, 1, tf.cast(new_height / shape[2], tf.int32), tf.cast(new_width / shape[3], tf.int32), 1])
    #     # output_shape=[shape[0], shape[1], new_height, new_width, shape[4]])
    tf.gradients(res, factor)
    return res


def instance_norm(activity):
    """Zscore the data."""
    mu, var = tf.nn.moments(activity, axes=[0, 1, 2, 3], keep_dims=True)
    std = tf.sqrt(var + 1e-8)
    activity = (activity - mu) / std
    return activity


def gaussian_kernel(
        size,
        mean,
        std):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        y = pass_through(y, dim=1)
        # y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
        #                  y.dtype)
        # y = tf.stop_gradient(y_hard - y) + y
    return y


def pass_through(y, dim=1):
    y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, dim, keep_dims=True)), y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
    return y


def differentiable_round(y):
    ceil = tf.ceil(y)
    return y - tf.stop_gradient(y - ceil)

# Note: this model was originally trained with conv3d layers initialized with
# TruncatedNormalInitializedVariable with stddev = 0.01.
def _predict_object_mask(input_patches, input_seed, fov_size, depth=9, is_training=True, adabn=False, min_rng=1., max_rng=1.1, step_size=1., im_mean=128., im_stddev=30.):
  """Computes single-object mask prediction."""

  in_k = 24
  ff_k = [24, 28, 32]
  ff_kpool_multiplier = 2
  learning = "hp_search"  # "regression"

  train_bn = True
  bn_decay = 0.95

  # Allow learned scaling
  # Init your scales, then softmax gumbel to select a scale

  if learning == "categorization":
    scales = np.arange(min_rng, max_rng, step_size).astype(np.float32)
    var_scale = tf.get_variable(
      name="scale_variable",
      dtype=tf.float32,
      trainable=False,
      initializer=scales)
    scale_selection = tf.get_variable(
      name="scale_selection",
      shape=len(scales),
      dtype=tf.float32,
      trainable=True,
      initializer=tf.initializers.zeros,  # tf.initializers.constant(np.zeros_like(scales) + (1. / len(scales))),
    )

    # Now gumbel selec
    tau = tf.get_variable(
      name="temperature",
      shape=(),
      dtype=tf.float32,
      trainable=False,
      initializer=tf.initializers.constant(1.)
    )  # Can be learned, but let's avoid for now
    scale_idx = gumbel_softmax(tf.reshape(scale_selection, [1, -1]), temperature=tau, hard=False)
    scale = tf.reduce_sum(var_scale * scale_idx)
    # scale_idx = tf.argmax(tf.reshape(scale_idx, [-1]))
    # scale = tf.gather(var_scale, scale_idx)  # var_scale[scale_idx]
  elif learning == "regression":
    var_scale = tf.get_variable(
      name="scale_variable",
      shape=(),
      dtype=tf.float32,
      trainable=True,
      initializer=tf.initializers.zeros,  # tf.initializers.constant(np.zeros_like(scales) + (1. / len(scales))),
    )
    # Transform to sigmoid then scale to min_rng/max_rng
    sig_scale = tf.sigmoid(var_scale)  # Init to 0.5
    assert max_rng > min_rng, "max_rng must be > min_rng"
    scale = (sig_scale + min_rng) * (max_rng - min_rng)
  elif learning == "hp_search":
    pass    
  else:
    raise NotImplementedError("{} not implemented".format(learning))

  # new_shape = tf.cast(scale * fov_size[:-1], tf.int32)

  if learning != "hp_search":
    res_patches = upsample_tf(input_patches, scale)
  else:
    res_patches = input_patches
  n, d, h, w, c = res_patches.get_shape().as_list()
  # new_shape = differentiable_round(scale * fov_size[:-1], dim=0)
  # res_patches = tf.image.resize_bilinear(
  #   tf.reshape(input_patches, [n * d, h, w, c]),
  #   new_shape,
  #   align_corners=True)
  # res_patches = tf.reshape(res_patches, [n, d, new_shape[0], new_shape[1], c])

  # test_sphere = generate_sphere_npy([d, h, w])
  # generate_sphere_npy([d, h, w])
  sphere = tf.stack([generate_sphere(tf.shape(res_patches)[1:-1]) for x in range(n)], 0)[..., None]
  # sphere = np.asarray([generate_sphere([d, h, w]) for _ in range(len(input_patches))])
  # sphere = generate_sphere([d, h, w])
  # input_patches = input_patches[..., 0]

  # sphere_input_patches = res_patches * sphere
  # image_mem = (sphere_input_patches - im_mean) / im_stddev
  image_mem = (res_patches - im_mean) / im_stddev

  # masked_images = tf.cast((input_patches * im_stddev) + im_mean, tf.int32)
  # masked_images = masked_images * tf.cast(sphere[None], tf.int32)
  # masked_images = masked_images * tf.cast(sphere, tf.int32)


  # Resume normal FFN
  image = tf.expand_dims(image_mem[..., 0], axis=4) * sphere
  membrane = tf.expand_dims(image_mem[..., 1], axis=4)
  image_k = in_k - 1

  x = layers.conv3d(image,
                                 trainable=False,
                                 scope='conv0_a',
                                 num_outputs=image_k,
                                 kernel_size=(1, 5, 5),
                                 padding='SAME')
  if input_patches.get_shape().as_list()[-1] == 2:
      print('FFN-hgru-v5: using membrane as input')
      x = tf.concat([x, membrane], axis=4)
  x = layers.conv3d(x,
                                 trainable=False,
                                 scope='conv0_b',
                                 num_outputs=x.get_shape().as_list()[-1],
                                 kernel_size=(1, 5, 5),
                                 padding='SAME')
  with tf.variable_scope('recurrent'):
      hgru_net = feedback_hgru_v5_3l_nu_f.hGRU(layer_name='hgru_net',
                                        num_in_feats=in_k,
                                        timesteps=1, #6, #8,
                                        h_repeat=1,
                                        hgru_dhw=[[1, 11, 11], [3, 5, 5], [5, 5, 5]],
                                        hgru_k=[in_k, ff_k[0], ff_k[1]],
                                        hgru_symmetric_weights=False,
                                        ff_conv_dhw=[[1, 7, 7], [1, 5, 5], [1, 5, 5]],
                                        ff_conv_k=ff_k,
                                        ff_kpool_multiplier=ff_kpool_multiplier,
                                        ff_pool_dhw=[[1, 2, 2], [2, 2, 2], [1, 2, 2]],
                                        ff_pool_strides=[[1, 2, 2], [2, 2, 2], [1, 2, 2]],
                                        fb_mode='transpose',
                                        fb_dhw=[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                                        fb_k=ff_k,
                                        padding='SAME',
                                        batch_norm=True,
                                        bn_reuse=False,
                                        gate_bn=True,
                                        aux=None,
                                        train=train_bn,
                                        bn_decay=bn_decay)

      # net = hgru_net.build(x, input_seed)
      l0, l1, l2 = hgru_net.build(x, input_seed)
  finalbn_param_initializer = {
      'moving_mean': tf.constant_initializer(0., dtype=tf.float32),
      'moving_variance': tf.constant_initializer(1., dtype=tf.float32),
      'gamma': tf.constant_initializer(0.1, dtype=tf.float32)
  }

  # Now take all the layers and concat them together
  # l0 = instance_norm(l0)  # hgru_net.ff0)
  # l1 = instance_norm(l1)  # hgru_net.ff1)
  # l2 = instance_norm(l2)  # hgru_net.ff2)

  l0_shape = tf.shape(l0)  # .get_shape()
  l1_shape = tf.shape(l1)  # .get_shape()
  l2_shape = tf.shape(l2)  # .get_shape()

  # l0 = tf.image.resize_bilinear(
  #   tf.reshape(l0, [l0_shape[0] * l0_shape[1], l0_shape[2], l0_shape[3], l0_shape[4]]),
  #   [h, w],
  #   align_corners=True)
  # l1 = tf.image.resize_bilinear(
  #   tf.reshape(l1, [l1_shape[0] * l1_shape[1], l1_shape[2], l1_shape[3], l1_shape[4]]),
  #   [h, w],
  #   align_corners=True)
  # l2 = tf.image.resize_bilinear(
  #   tf.reshape(l2, [l2_shape[0] * l2_shape[1], l2_shape[2], l2_shape[3], l2_shape[4]]),
  #   [h, w],
  #   align_corners=True)
  # l0 = tf.reshape(l0, [n, d, h, w, ])
  # l1 = tf.reshape(l1, [n, d, h, w, ff_k[0]])
  # l2 = tf.reshape(l2, [n, d, h, w, ff_k[2]])
  # net = tf.concat([l0, l1, l2], -1)
  # sphere_input_patchesi
  # net = tf.concat([l0, l1], -1)
  net = l0
  net = instance_norm(net)
  logits = layers.conv3d(net,
                                    scope='filling_in_0',
                                    num_outputs=255,
                                    kernel_size=(1, 5, 5),
                                    activation_fn=tf.nn.relu)
  logits = layers.conv3d(logits,
                                    scope='filling_in_1',
                                    num_outputs=255,  # CCE intensity per pixel
                                    kernel_size=(1, 1, 1),
                                    activation_fn=None)

  # Now create the loss
  flip_sphere = tf.squeeze(1 - sphere, -1)
  image_label = tf.cast(res_patches[..., 0] * flip_sphere, tf.int32)
  logits = logits * flip_sphere[..., None]

  loss = tf.losses.sparse_softmax_cross_entropy(image_label, logits, reduction=tf.losses.Reduction.NONE)
  loss = tf.reduce_mean(loss * flip_sphere)

  # Now create the optimizer
  optim = tf.train.AdamOptimizer(3e-4)
  train_op = optim.minimize(loss)
  # tf.gradients(loss, net)
  # scale_grad = tf.gradients(loss, var_scale)[0]

  ops = {
    "train_op": train_op,
    "loss": loss,
    # "images": input_patches,
    # "scale_grad": scale_grad,
    "image_label": image_label,
    "logits": logits,
    # "scale": scale
  }
  return ops  # , input_patches


def generate_sphere(volumeSize, div=4, r=150):
    x_ = tf.linspace(0,volumeSize[0], volumeSize[0])
    y_ = tf.linspace(0,volumeSize[1], volumeSize[1])
    z_ = tf.linspace(0,volumeSize[2], volumeSize[2])
    # center = r = [v/2 for v in volumeSize] # radius can be changed by changing r value
    volumeSizeBounds = tf.cast(volumeSize / div, tf.float32)
     
    # minSize = tf.cast(volumeSize / (div + 1), tf.float32)
    # r = [5., 15., 15.]  # tf.maximum(tf.cast(tf.random.uniform([3]) * volumeSizeBounds, tf.float32), minSize)
    # c = [15., 30., 30.]  # tf.maximum(tf.cast(tf.random.uniform([3]) * volumeSizeBounds, tf.float32), minSize)
    # maxpos = tf.cast(volumeSize, tf.float32) - volumeSizeBounds
    minpos = [12., 18., 18.]
    maxpos = tf.cast(volumeSize, tf.float32) - minpos
    c = (tf.random.uniform(shape=[3], minval=0, maxval=1) * maxpos) + minpos

    u, v, w = tf.meshgrid(x_, y_, z_, indexing='ij')
    u = tf.cast(u, tf.float32)
    v = tf.cast(v, tf.float32)
    w = tf.cast(w, tf.float32)
    a = tf.pow(u-c[0], 2)+tf.pow(v-c[1], 2)+tf.pow(w-c[2], 2)
    a = tf.cast(tf.greater(a, r), tf.float32)
    return a


def generate_sphere_npy(volumeSize, div=4):
    x_ = np.linspace(0,volumeSize[0], volumeSize[0])
    y_ = np.linspace(0,volumeSize[1], volumeSize[1])
    z_ = np.linspace(0,volumeSize[2], volumeSize[2])
    # center = r = [v/2 for v in volumeSize] # radius can be changed by changing r value
    volumeSizeBounds = np.asarray([x / div for x in volumeSize])
    r = 20  # np.asarray([5., 30., 30.])  # tf.maximum(tf.cast(tf.random.uniform([3]) * volumeSizeBounds, tf.float32), minSize)
    c = np.asarray([15., 30., 30.])  # tf.maximum(tf.cast(tf.random.uniform([3]) * volumeSizeBounds, tf.float32), minSize)
    maxpos = volumeSize - volumeSizeBounds
    c = (np.random.uniform(size=3) * maxpos) + (volumeSizeBounds / 2)

    u,v,w = np.meshgrid(x_, y_, z_, indexing='ij')
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    w = w.astype(np.float32)
    a = (u - c[0]) ** 2 + (v - c[1]) ** 2 + (w - c[2]) ** 2
    a = (a < r.min()).astype(np.float32)
    return a


# def conv_blur(kernel=[]):


class ConvStack3DFFNModel(model.FFNModel):
  dim = 3

  def __init__(self, with_membrane=False, fov_size=None, optional_output_size=None, deltas=None, batch_size=None, depth=9,
               is_training=True, adabn=False, reuse=False, tag='', TA=None, grad_clip_val=0.0, im_mean=128., im_stddev=30.):
    super(ConvStack3DFFNModel, self).__init__(deltas, batch_size, with_membrane, validation_mode=not(is_training), tag=tag)

    self.optional_output_size = optional_output_size
    self.set_uniform_io_size(fov_size)
    self.depth = depth
    self.reuse=reuse
    self.TA=TA
    self.is_training=is_training
    self.adabn=adabn
    self.im_mean = im_mean
    self.im_stddev = im_stddev
    if grad_clip_val is None:
        self.grad_clip_val = 0.0
    else:
        self.grad_clip_val = grad_clip_val

  def __call__(self):

    if self.input_patches is None:
      self.input_patches = tf.placeholder(
          tf.float32, [1] + list(self.input_image_size[::-1]) +[1],
          name='patches')

    with tf.variable_scope('seed_update', reuse=self.reuse):
      self.ops = _predict_object_mask(self.input_patches, self.input_seed, self.input_image_size,
                                          depth=self.depth, is_training=self.is_training, adabn=self.adabn, im_mean=self.im_mean, im_stddev=self.im_stddev)
