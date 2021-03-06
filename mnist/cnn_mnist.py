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
  # 1次元化(フラットにする)
  # [batch_size, 1024]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  # 分類できるようにdense層に繋げる
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  # ドロップアウト:
  # A Simple Way to Prevent Neural Networks from Overfitting
  # ある更新で層の中のノードのうちのいくつかを無効にして（そもそも存在しないかのように扱って）学習を行い、
  # 次の更新では別のノードを無効にして学習を行うことを繰り返します。
  # これにより学習時にネットワークの自由度を強制的に小さくして汎化性能を上げ、
  # 過学習を避けることができます。隠れ層においては、一般的に50%程度を無効すると良いと言われています。
  # @see: http://sonickun.hatenablog.com/entry/2016/07/18/191656
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer #2(Logits Layer): 10 neurons, one for each digit target class (0–9)
  # 最後のレイヤーで、返り値を結果にする
  # [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1), # 一番大きい値のindexを返す
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

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
