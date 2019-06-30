# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# import sys
# sys.setrecursionlimit(1000000)

x = tf.constant([1, 2, 3, 4, 5, 6], tf.float32)
y = tf.constant([3, 4, 7, 8, 11, 14], tf.float32)

w = tf.Variable(1.0, dtype=tf.float32)
b = tf.Variable(1.0, dtype=tf.float32)

loss = tf.reduce_sum(tf.square(y - (w * x + b)))
session = tf.Session()
session.run(tf.global_variables_initializer())

opti = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
MSE = []
for i in range(500):
	session.run(opti)
	MSE.append(session.run(loss))

	if i % 50 == 0:
		print((session.run(w), session.run(b)))

plt.figure(1)
plt.plot(MSE)
plt.show()

plt.figure(2)
x_array, y_array = session.run([x, y])
plt.plot(x_array, y_array, 'o')

xx = np.arange(0, 10, 0.05)
w_arr = session.run(w)
b_arr = session.run(b)
yy = float(w_arr) * xx + float(b_arr)
plt.plot(xx, yy)
plt.show()