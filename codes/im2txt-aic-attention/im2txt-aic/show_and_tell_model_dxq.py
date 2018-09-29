# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import image_embedding
import image_processing
import inputs as input_ops
#from im2txt.ops import image_embedding
#from im2txt.ops import image_processing
#from im2txt.ops import inputs as input_ops


class ShowAndTellModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

    # Setup initializer
    self.weight_initializer = tf.contrib.layers.xavier_initializer()
    self.const_initializer = tf.constant_initializer(0.0)

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])

      # Batch inputs.
      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)
      images, input_seqs, target_seqs, input_mask = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings

    Modidication: Jay
        provide image_embeddings with [batch_size, 64, embedding_size]
    """
    inception_output = image_embedding.inception_resnet_v2(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    print('2',inception_output.get_shape())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionResnetV2")
    print('inception_output',inception_output)
    spatial_dim = inception_output.get_shape()[1].value*inception_output.get_shape()[2].value
    print('spatial_dim',spatial_dim)
    channel_dim = inception_output.get_shape()[3].value
    print('channel_dim',channel_dim)
    image_features = tf.reshape(inception_output, [-1, channel_dim])
    print('image_features',image_features)

    # Map inception output into embedding space [batch_size, spatial_dim, embedding_size].
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=image_features,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    # reshape into [batch_size, spatial_dim, embedding_size]
    image_embeddings = tf.reshape(image_embeddings, [-1, spatial_dim, self.config.embedding_size])
    print('image_embeddings',image_embeddings)
    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings
    # self.image_embeddings = tf.reshape(image_embeddings, [-1, 64, 1536])  # Keep the spatial order information

  def get_initial_lstm(self, features):
      with tf.variable_scope('initial_lstm'):
          features_mean = tf.reduce_mean(features, 1)
          H = self.config.num_lstm_units
          D = tf.shape(features)[2]
          w_h = tf.get_variable('w_h', [D, H], initializer=self.weight_initializer)
          b_h = tf.get_variable('b_h', [H], initializer=self.const_initializer)
          h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

          w_c = tf.get_variable('w_c', [D, H], initializer=self.weight_initializer)
          b_c = tf.get_variable('b_c', [H], initializer=self.const_initializer)
          c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
          return c, h

  def project_features(self, features):
    with tf.variable_scope('project_features'):
      L = tf.shape(features)[1]
      D = tf.shape(features)[2]
      w = tf.get_variable('w', [D, D], initializer=self.weight_initializer)
      features_flat = tf.reshape(features, [-1, D])
      features_proj = tf.matmul(features_flat, w)
      features_proj = tf.reshape(features_proj, [-1, L, D])
      return features_proj

  def attention_layer(self, features, features_proj, h, reuse=False):
      """ Calculation of attention layer
      :param features:
      :param features_proj:
      :param h:
      :param reuse:
      :return:
      """
      with tf.variable_scope('attention_layer', reuse=reuse):
          H = self.config.num_lstm_units
          D = tf.shape(features)[2]
          w = tf.get_variable('w', [H, D], initializer=self.weight_initializer)
          b = tf.get_variable('b', [D], initializer=self.const_initializer)
          w_att = tf.get_variable('w_att', [D, 1], initializer=self.weight_initializer)

          h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D); tf.expand_dims used for dimension matching
          out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
          alpha = tf.nn.softmax(out_att)
          context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
          return context, alpha

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

    self.seq_embeddings = seq_embeddings

  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    # Modification: Incorporation of attention mode

    # Definition of LSTM cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    if self.mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      # image_embeddings is the context features from images.

      '''zero_state = lstm_cell.zero_state(
          batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
      _, (c, h) = lstm_cell(inputs=tf.concat([self.image_embeddings, self.image_embeddings], 1), state=zero_state)'''

      # calculation of c and h for LSTM_cell
      c, h = self.get_initial_lstm(features=self.image_embeddings)
      initial_state = (c, h)
      features_proj = self.project_features(features=self.image_embeddings.as_list())  # calculation for image features

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()

      if self.mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        tf.concat(axis=1, values=initial_state, name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
            state=state_tuple)

        # Concatentate the resulting state.
        tf.concat(axis=1, values=state_tuple, name="state")
      else:
        # Run the batch of sequence embeddings through the LSTM.
        # Jay--Section for training; Using for loop to substitute dynamic_rnn
        sequence_length = tf.reduce_sum(self.input_mask, 1)
        '''lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=self.seq_embeddings,
                                            sequence_length=sequence_length,
                                            initial_state=initial_state,
                                            dtype=tf.float32,
                                            scope=lstm_scope)'''
        # Jay--Modification: Every step update and use attention efficient
        lstm_outputs = []
        for time_step in range(sequence_length):
            # Use attention_layer to calculate context vectors for attention mode
            context, _ = self.attention_layer(self.image_embeddings, features_proj, h)  # Calcultion of attention coefficients
            cell_output, (c, h) = lstm_cell(inputs=tf.concat([self.seq_embeddings[:, time_step, :]\
                , context], 1), state=[c, h])  # Get the hidden_state parameter for attention_mode calculation
            lstm_outputs.append(cell_output)

    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)

    if self.mode == "inference":
      tf.nn.softmax(logits, name="softmax")
    else:
      targets = tf.reshape(self.target_seqs, [-1])
      weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

      # Compute losses.
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                              logits=logits)
      batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                          tf.reduce_sum(weights),
                          name="batch_loss")
      tf.losses.add_loss(batch_loss)
      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss
      self.target_cross_entropy_losses = losses  # Used in evaluation.
      self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()
