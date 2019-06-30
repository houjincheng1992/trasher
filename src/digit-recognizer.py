# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

train_filename = "../kaggle_data/digit-recognizer/train.csv"
test_filename = "../kaggle_data/digit-recognizer/test.csv"


def load_csv_data(filename, header, session):
    """
        return tensorflow tensor
    """
    numpy_data = pd.read_csv(filename, header=header).values
    # print(type(numpy_data))
    # print(numpy_data)
    # print("here")
    # data_tensor = tf.convert_to_tensor(numpy_data, tf.int32, name="data_tensor")
    # print(data_tensor)
    # session = tf.compat.v1.Session()
    # print(data_tensor.eval(session=session))
    # session = tf.compat.v1.Session()
    # print(data_tensor)
    # print("here")
    return numpy_data


def train_model(labels, features, session):
    """
        model func
    """
    I, H1, O = 784, 200, 10

    x = tf.placeholder(tf.float32, [None, I])
    y = tf.placeholder(tf.float32, [None, O])
    w1 = tf.Variable(tf.random.normal([I, H1], 0, 1, tf.float32), dtype=tf.float32, name="w1")
    b1 = tf.Variable(tf.random.normal([H1], 0, 1, tf.float32), dtype=tf.float32, name="b1")
    l1 = tf.matmul(x, w1) + b1
    sigma1 = tf.nn.sigmoid(l1)

    w2 = tf.Variable(tf.random.normal([H1, O], 0, 1, tf.float32), dtype=tf.float32, name="w2")
    b2 = tf.Variable(tf.random.normal([O], 0, 1, tf.float32), dtype=tf.float32, name="b2")
    # y = tf.matmul(sigma1, w2) + b2

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=tf.matmul(sigma1, w2) + b2))
    opti = tf.compat.v1.train.AdamOptimizer(0.001, 0.9, 0.999, 1e-8).minimize(loss)

    session.run(tf.compat.v1.global_variables_initializer())
    session.run(tf.compat.v1.local_variables_initializer())
    LOSS = []

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    print(features.shape)
    print(labels.shape)
    for i in range(20):
        LOSS.append(session.run(loss, feed_dict={x: features.astype(float), y: labels.astype(float)}))
        session.run(opti, feed_dict={x: features, y: labels})
    coord.request_stop()
    coord.join(threads)

    plt.figure(1)
    plt.plot(LOSS)
    plt.show()



def process_train(train_data, session):
    """
        train main func
    """
    # shape = tf.shape(train_data)
    # [width, height] = session.run(shape)
    # print(width)
    # print(height)

    # one-hot encode
    labels = np.eye(10)[train_data[:, 0]]
    features = train_data[:, 1: ]
    # labels_pre = tf.slice(train_data, [0, 0], [int(width), 1])
    # features_pre = tf.slice(train_data, [0, 1], [int(width), int(height) - 1])
    # labels = tf.one_hot(labels_pre, 10, dtype=tf.float32)
    # features = tf.cast(tf.cast(features_pre, tf.bool), tf.float32)
    # print(session.run(label).shape)
    # print(session.run(features).shape)
    train_model(labels, features, session)
    return

if __name__ == '__main__':
    session = tf.compat.v1.Session()
    # load train data
    train_data = load_csv_data(train_filename, 0, session)
    process_train(train_data, session)