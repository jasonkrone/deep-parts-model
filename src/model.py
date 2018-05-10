# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
"""Model using memory component.

The model embeds images using a standard CNN architecture.
These embeddings are used as keys to the memory component,
which returns nearest neighbors.
"""

import tensorflow as tf
import numpy as np

import memory

# from models repo
import sys
sys.path.append('/home/jason/models/research/slim/')
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
from nets import resnet_v2
from tensorflow.contrib import slim

FLAGS = tf.flags.FLAGS


class BasicClassifier(object):

  def __init__(self, output_dim):
    self.output_dim = output_dim

  def core_builder(self, memory_val, x, y):
    del x, y
    y_pred = memory_val
    loss = 0.0

    return loss, y_pred


class LeNet(object):
  """Standard CNN architecture."""

  def __init__(self, image_size, num_channels, hidden_dim):
    self.image_size = image_size
    self.num_channels = num_channels
    self.hidden_dim = hidden_dim
    self.matrix_init = tf.truncated_normal_initializer(stddev=0.1)
    self.vector_init = tf.constant_initializer(0.0)

  def core_builder(self, x, is_training):
    """Embeds x using standard CNN architecture.

    Args:
      x: Batch of images as a 2-d Tensor [batch_size, -1].

    Returns:
      A 2-d Tensor [batch_size, hidden_dim] of embedded images.
    """

    ch1 = 64  # number of channels in 1st layer
    ch2 = 64  # number of channels in 2nd layer
    ch3 = 64  # number of channels in 3rd layer
    ch4 = 64  # number of channels in 4th layer

    with tf.variable_scope('embedder', reuse=tf.AUTO_REUSE):
      conv1_weights = tf.get_variable('conv1_w',
                                    [3, 3, self.num_channels, ch1],
                                    initializer=self.matrix_init)
      conv1_biases = tf.get_variable('conv1_b', [ch1],
                                   initializer=self.vector_init)

      conv2_weights = tf.get_variable('conv2_w', [3, 3, ch1, ch2],
                                    initializer=self.matrix_init)
      conv2_biases = tf.get_variable('conv2_b', [ch2],
                                   initializer=self.vector_init)

      conv3_weights = tf.get_variable('conv3_w', [3, 3, ch2, ch3],
                                    initializer=self.matrix_init)
      conv3_biases = tf.get_variable('conv3_b', [ch3],
                                   initializer=self.vector_init)

      conv4_weights = tf.get_variable('conv4_w', [3, 3, ch3, ch4],
                                    initializer=self.matrix_init)
      conv4_biases = tf.get_variable('conv4_b', [ch4],
                                   initializer=self.vector_init)

      # fully connected
      fc1_weights = tf.get_variable(
        'fc1_w', [np.ceil(self.image_size / 16.0) * np.ceil(self.image_size / 16.0) * ch4,
                  self.hidden_dim], initializer=self.matrix_init)
      fc1_biases = tf.get_variable('fc1_b', [self.hidden_dim],
                                 initializer=self.vector_init)

    # define model
    #x = tf.reshape(x,[-1, self.image_size, self.image_size, self.num_channels])
    batch_size = tf.shape(x)[0]
    conv1 = tf.nn.conv2d(x, conv1_weights,
                         strides=[1, 1, 1, 1], padding='SAME')
    conv1=tf.contrib.layers.batch_norm(conv1, center=True, 
                                      is_training=is_training)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.conv2d(pool1, conv2_weights,
                         strides=[1, 1, 1, 1], padding='SAME')
    conv2=tf.contrib.layers.batch_norm(conv2, center=True, 
                                      is_training=is_training)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv3 = tf.nn.conv2d(pool2, conv3_weights,
                         strides=[1, 1, 1, 1], padding='SAME')
    conv3=tf.contrib.layers.batch_norm(conv3, center=True, 
                                      is_training=is_training)
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    conv4 = tf.nn.conv2d(pool3, conv4_weights,
                         strides=[1, 1, 1, 1], padding='SAME')
    conv4=tf.contrib.layers.batch_norm(conv4, center=True, 
                                      is_training=is_training)
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
    pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

    reshape = tf.reshape(pool4, [batch_size, -1])
    hidden = tf.matmul(reshape, fc1_weights) + fc1_biases

    return hidden

class Resnet50(object):
  def __init__(self, image_size, num_channels, hidden_dim):
    self.image_size = image_size
    self.num_channels = num_channels
    self.hidden_dim = hidden_dim

  def core_builder(self, x, is_training, reuse=False):
    batch_size = tf.shape(x)[0]
    with tf.contrib.framework.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_50(x, num_classes=None, is_training=is_training, reuse=reuse)
        features = end_points['global_pool'] # batch_size, 1, 1, 2048
    out = tf.reshape(features, [batch_size, -1])
    return out

class InceptionV3(object):

  def __init__(self, image_size, num_channels, hidden_dim):
    self.image_size = image_size
    self.num_channels = num_channels
    self.hidden_dim = hidden_dim

  def core_builder(self, x, is_training, reuse=False):
    # these layers are frozen
    batch_size = tf.shape(x)[0]
    with tf.contrib.framework.arg_scope(inception_v3_arg_scope()):
      # this expects 299 dim input
      logits, end_points = inception_v3(x, num_classes=None, is_training=is_training, reuse=reuse, create_aux_logits=False)
      # 8x8x2048
      features = end_points['Mixed_7c']
    out = tf.reshape(features, [batch_size, -1])
    #  conv_flat = tf.reshape(conv, [batch_size, 8*8*128])
    # these layers are trained
    #with tf.variable_scope('added_layers', reuse=reuse):
    # 1 x 1 conv with 128 channels
    #  conv = tf.layers.conv2d(inputs=features, filters=128, kernel_size=[1, 1], padding="same", activation=tf.nn.relu, name='added_1x1conv')
    #  conv_flat = tf.reshape(conv, [batch_size, 8*8*128])
    #  out = tf.layers.dense(inputs=conv_flat, units=self.hidden_dim, activation=tf.nn.relu, name='added_fc')
    return out

class Model(object):
  """Model for coordinating between CNN embedder and Memory module."""

  def __init__(self, input_dim, output_dim, rep_dim, memory_size, vocab_size,
               learning_rate=0.0001, use_lsh=False, episode_length=None):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.rep_dim = rep_dim
    self.memory_size = memory_size
    self.vocab_size = vocab_size
    self.learning_rate = learning_rate
    self.use_lsh = use_lsh
    self.episode_length = episode_length

    #self.embedder = self.get_embedder()
    self.embedder = self.get_embedder_with_parts()
    self.memory = self.get_memory()
    self.classifier = self.get_classifier()

    self.global_step = tf.train.get_or_create_global_step()

  def get_embedder_with_parts(self):
    h, w, c = self.input_dim
    #return LeNet(int(h), c, self.rep_dim)
    if FLAGS.use_resnet:
        return Resnet50(int(h), c, self.rep_dim)
    else:
        return InceptionV3(int(h), c, self.rep_dim)

  def get_embedder(self):
    return LeNet(int(self.input_dim ** 0.5), 1, self.rep_dim)

  def get_memory(self):
    cls = memory.LSHMemory if self.use_lsh else memory.Memory
    return cls(self.rep_dim, self.memory_size, self.vocab_size)

  def get_classifier(self):
    return BasicClassifier(self.output_dim)

  def core_builder_with_parts(self, x, p1, p2, y, keep_prob, use_recent_idx=True):
    #embeddings = self.embedder.core_builder(x)
    embeddings_p1 = self.embedder.core_builder(p1, self.is_training)
    embeddings_p2 = self.embedder.core_builder(p2, self.is_training, reuse=True)
    # just treat the part embeddings as extra examples
    embeddings = tf.concat([embeddings_p1, embeddings_p2], 0)
    # duplicate the y values so you have the right number
    if keep_prob < 1.0:
      embeddings = tf.nn.dropout(embeddings, keep_prob)
    memory_val, _, teacher_loss = self.memory.query(
        embeddings, y, use_recent_idx=use_recent_idx)
    # TODO: this classification doesn't vote
    loss, y_pred = self.classifier.core_builder(memory_val, x, y)
    return loss + teacher_loss, y_pred

  def core_builder(self, x, y, keep_prob, use_recent_idx=True):
    embeddings = self.embedder.core_builder(x, self.is_training)
    if keep_prob < 1.0:
      embeddings = tf.nn.dropout(embeddings, keep_prob)
    memory_val, _, teacher_loss = self.memory.query(
        embeddings, y, use_recent_idx=use_recent_idx)
    loss, y_pred = self.classifier.core_builder(memory_val, x, y)

    return loss + teacher_loss, y_pred

  def train_with_parts(self, x, p1, p2, y):
    loss, _ = self.core_builder_with_parts(x, p1, p2, y, keep_prob=0.3)
    gradient_ops = self.training_ops(loss)
    return loss, gradient_ops

  def train(self, x, y):
    loss, _ = self.core_builder(x, y, keep_prob=0.3)
    gradient_ops = self.training_ops(loss)
    return loss, gradient_ops

  def eval_with_parts(self, x, p1, p2, y):
    _, y_preds = self.core_builder_with_parts(x, p1, p2, y, keep_prob=1.0, use_recent_idx=False)
    return y_preds

  def eval(self, x, y):
    _, y_preds = self.core_builder(x, y, keep_prob=1.0,
                                   use_recent_idx=False)
    return y_preds

  def get_placeholders_with_parts(self):
    x_ph = tf.placeholder(tf.float32, [None] + list(self.input_dim))
    p1_ph = tf.placeholder(tf.float32, [None] + list(self.input_dim))
    p2_ph = tf.placeholder(tf.float32, [None] + list(self.input_dim))
    y_ph = tf.placeholder(tf.int32, [None])
    return (x_ph, p1_ph, p2_ph, y_ph)

  def get_xy_placeholders(self):
    shape = [None] + list(self.input_dim)
    return (tf.placeholder(tf.float32, shape),
            tf.placeholder(tf.int32, [None]))

  def setup(self):
    """Sets up all components of the computation graph."""

    #self.x, self.y = self.get_xy_placeholders()
    self.is_training = tf.placeholder(tf.bool)
    self.x, self.p1, self.p2, self.y = self.get_placeholders_with_parts()

    # This context creates variables
    with tf.variable_scope('core', reuse=None):
      #self.loss, self.gradient_ops = self.train(self.x, self.y)
      self.loss, self.gradient_ops = self.train_with_parts(self.x, self.p1, self.p2, self.y)
    # And this one re-uses them (thus the `reuse=True`)
    with tf.variable_scope('core', reuse=True):
      #self.y_preds = self.eval(self.x, self.y)
      self.y_preds = self.eval_with_parts(self.x, self.p1, self.p2, self.y)

  def training_ops(self, loss):
    opt = self.get_optimizer()
    params = tf.trainable_variables()
    # do not train the features from inception or resnet
    params = [v for v in params if ('InceptionV3' not in v.name) and ('resnet_v2' not in v.name)]
    print('training params:', params)
    gradients = tf.gradients(loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    return opt.apply_gradients(zip(clipped_gradients, params),
                               global_step=self.global_step)

  def get_optimizer(self):
    return tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                  epsilon=1e-4)

  def one_step_with_parts(self, sess, x, p1, p2, y, is_training):
    outputs = [self.loss, self.gradient_ops]
    return sess.run(outputs, feed_dict={self.x : x, self.p1 : p1, self.p2 : p2, self.y : y, self.is_training : is_training})

  def one_step(self, sess, x, y):
    outputs = [self.loss, self.gradient_ops]
    return sess.run(outputs, feed_dict={self.x: x, self.y: y})

  def episode_step_with_parts(self, sess, x, p1, p2, y, is_training, clear_memory=False):   
    outputs = [self.loss, self.gradient_ops]

    if clear_memory:
      self.clear_memory(sess)

    losses = []
    for xx, pp1, pp2, yy in zip(x, p1, p2, y):
      out = sess.run(outputs, feed_dict={self.x: xx, self.p1 : pp1, self.p2 : pp2, self.y: yy, self.is_training : is_training})
      loss = out[0]
      losses.append(loss)

    return losses

  # TODO: create placeholder here for parts
  def episode_step(self, sess, x, y, clear_memory=False):
    """Performs training steps on episodic input.

    Args:
      sess: A Tensorflow Session.
      x: A list of batches of images defining the episode.
      y: A list of batches of labels corresponding to x.
      clear_memory: Whether to clear the memory before the episode.

    Returns:
      List of losses the same length as the episode.
    """

    outputs = [self.loss, self.gradient_ops]

    if clear_memory:
      self.clear_memory(sess)

    losses = []
    for xx, yy in zip(x, y):
      out = sess.run(outputs, feed_dict={self.x: xx, self.y: yy, self.is_training : True})
      loss = out[0]
      losses.append(loss)

    return losses

  def predict(self, sess, x, y=None):
    """Predict the labels on a single batch of examples.

    Args:
      sess: A Tensorflow Session.
      x: A batch of images.
      y: The labels for the images in x.
        This allows for updating the memory.

    Returns:
      Predicted y.
    """

    # Storing current memory state to restore it after prediction
    mem_keys, mem_vals, mem_age, _ = self.memory.get()
    cur_memory = (
        tf.identity(mem_keys),
        tf.identity(mem_vals),
        tf.identity(mem_age),
        None,
    )

    outputs = [self.y_preds]
    if y is None:
      ret = sess.run(outputs, feed_dict={self.x: x})
    else:
      ret = sess.run(outputs, feed_dict={self.x: x, self.y: y})

    # Restoring memory state
    self.memory.set(*cur_memory)

    return ret

  def episode_predict_with_parts(self, sess, x, p1, p2, y, is_training, clear_memory=False):
    """Predict the labels on an episode of examples.

    Args:
      sess: A Tensorflow Session.
      x: A list of batches of images.
      y: A list of labels for the images in x.
        This allows for updating the memory.
      clear_memory: Whether to clear the memory before the episode.

    Returns:
      List of predicted y.
    """

    # Storing current memory state to restore it after prediction
    mem_keys, mem_vals, mem_age, _ = self.memory.get()
    cur_memory = (
        tf.identity(mem_keys),
        tf.identity(mem_vals),
        tf.identity(mem_age),
        None,
    )

    if clear_memory:
      self.clear_memory(sess)

    outputs = [self.y_preds]
    y_preds = []
    for xx, pp1, pp2, yy in zip(x, p1, p2, y):
      out = sess.run(outputs, feed_dict={self.x: xx, self.p1 : pp1, self.p2 : pp2, self.y: yy, self.is_training : False})
      y_pred = out[0]
      y_preds.append(y_pred)

    # Restoring memory state
    self.memory.set(*cur_memory)

    return y_preds


  def episode_predict(self, sess, x, y, clear_memory=False):
    """Predict the labels on an episode of examples.

    Args:
      sess: A Tensorflow Session.
      x: A list of batches of images.
      y: A list of labels for the images in x.
        This allows for updating the memory.
      clear_memory: Whether to clear the memory before the episode.

    Returns:
      List of predicted y.
    """

    # Storing current memory state to restore it after prediction
    mem_keys, mem_vals, mem_age, _ = self.memory.get()
    cur_memory = (
        tf.identity(mem_keys),
        tf.identity(mem_vals),
        tf.identity(mem_age),
        None,
    )

    if clear_memory:
      self.clear_memory(sess)

    outputs = [self.y_preds]
    y_preds = []
    for xx, yy in zip(x, y):
      out = sess.run(outputs, feed_dict={self.x: xx, self.y: yy, self.is_training : False})
      y_pred = out[0]
      y_preds.append(y_pred)

    # Restoring memory state
    self.memory.set(*cur_memory)

    return y_preds

  def clear_memory(self, sess):
    sess.run([self.memory.clear()])
