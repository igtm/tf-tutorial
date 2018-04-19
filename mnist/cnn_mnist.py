from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # 第一引数の Tensor<特徴> を 指定の形に reshapeする
  # batch_size: -1(-1は features['x'] の数によって動的に計算されるということ。ここのbatch_sizeは hyperparameterで、チューニングできる要素の一つ。), image_width: 28, image_height: 28, channnels: 1(black: 色の数のこと)
  # ex) batch_sizeが5: [5, 28, 28, 1] => 5 * 28 * 28 = 3920個の値が features['x']に含まれるということになる
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
  # CNNの畳み込みに関しては これがわかりやすい @see: https://qiita.com/icoxfog417/items/5fd55fad152231d706c2
  # 畳み込み層: 特徴抽出する意味がある
  # => [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32, # filterの数 (2の累乗をセットすることが多い。32,64,128...)
      kernel_size=[5, 5], # filterの大きさ/サイズ
      padding="same", # 余白部分: valid(default) or same: sameはoutputがinputと同じwidth,heightになるようにpaddingを自動で挿入してくれる。普通にpadding無しでやると 28x28 5x5 => 24x24になってしまう。(inputsize-filtersize)/stride + 1   (28-5)/1 + 1 = 24
      activation=tf.nn.relu)

  # Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
  # プーリング層: データ圧縮の意味がある
  # maxpooling: filter内の最大値を取っていく手法
  # => [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2) # [14,14] の半分になる

  # Convolutional Layer #2 and Pooling Layer #2: Applies 64 5x5 filters, with ReLU activation function
  # => [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
      
  # => [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2) # Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2

  # Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element will be dropped during training)
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer #2(Logits Layer): 10 neurons, one for each digit target class (0–9)
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
  tf.app.run()
