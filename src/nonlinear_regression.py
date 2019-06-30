# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None])
y = tf.placeholder(tf.float32, [None])
z = tf.placeholder(tf.float32, [None])

w1 = tf.Variable(initial_value = 2.0, dtype = tf.float32, name = 'w1')
w2 = tf.Variable(initial_value = 2.0, dtype = tf.float32, name = 'w2')

loss = tf.reduce_sum(tf.square(z - tf.pow((w1 * x + w2 * y), 2.0)))
opti = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

x_train = np.array([1, 2, 3, 1, 4, 5], np.float32)
y_train = np.array([1, 1, 2, 2, 5, 8], np.float32)
z_train = np.array([8, 12, 10, 14, 28, 10], np.float32)

session = tf.Session()
for i in range(10):
    session.run(opti, feed_dict={x: x_train, y: y_train, z: z_train})
print("done")