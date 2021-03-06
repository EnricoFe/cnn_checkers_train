# Title: parser.py
# Author: Chris Larson
# CS-6700 Final Project
# This program trains a convnet using checkers games extracted from 'OCA_2.0.pdn'.

# Comments ========================================================================================
#
# Board positions:
#  00 01 02 03
#  04 05 06 07
#  08 09 10 11
#  12 13 14 15
#  16 17 18 19
#  20 21 22 23
#  24 25 26 27
#  28 29 30 31
#
#
# Initial board:
#  1  1  1  1
#  1  1  1  1
#  1  1  1  1
#  0  0  0  0
#  0  0  0  0
# -1 -1 -1 -1
# -1 -1 -1 -1
# -1 -1 -1 -1
#
#
# Label output:
#   R L U D
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#   0 0 0 0
#
# end comments ====================================================================================

import os
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle
import tensorflow as tf
import read_pickle as rp
import time


def accuracy(preds, labs):
    n = preds.shape[0]
    acc_vec = np.ndarray([n, 1])
    for i in range(n):
        acc_vec[i, 0] = sum(preds[i, :].astype(int) * labs[i, :].astype(int))
        # print(acc_vec[i, 0])
        assert acc_vec[i, 0] in range(0, 6)
    acc_score = list()
    for j in range(1, 6):
        percentage = len(np.argwhere(acc_vec == j)) / float(n)
        acc_score.append(round(percentage, 4))
    return acc_score


def deepnet(num_steps, lambda_loss, dropout_L1, dropout_L2, ckpt_dir):

    # Computational graph
    graph = tf.Graph()
    with graph.as_default():

        # Inputs
        tf_xTr = tf.placeholder(tf.float32, shape=[batch_size, board_height * board_width])
        tf_yTr = tf.placeholder(tf.float32, shape=[batch_size, label_height * label_width])
        tf_xTe = tf.constant(xTe)
        tf_xTr_full = tf.constant(xTr)

        # Variables
        w1 = tf.Variable(tf.truncated_normal([board_height * board_width, num_nodes_layer1], stddev=0.1))
        b1 = tf.Variable(tf.zeros([num_nodes_layer1]))
        w2 = tf.Variable(tf.truncated_normal([num_nodes_layer1, num_nodes_layer2], stddev=0.1))
        b2 = tf.Variable(tf.zeros([num_nodes_layer2]))
        w3 = tf.Variable(tf.truncated_normal([num_nodes_layer2, label_height * label_width], stddev=0.1))
        b3 = tf.Variable(tf.zeros([label_height * label_width]))

        # Train
        def model(xTr_batch, dropout_switch):
            y1 = tf.add(tf.matmul(xTr_batch, w1), b1)
            h1 = tf.nn.relu(y1)
            h1_out = tf.nn.dropout(h1, 1 - dropout_L1 * dropout_switch)
            y2 = tf.add(tf.matmul(h1_out, w2), b2)
            h2 = tf.nn.relu(y2)
            y2_out = tf.nn.dropout(h2, 1 - dropout_L2 * dropout_switch)
            return tf.add(tf.matmul(y2_out, w3), b3)

        logits = model(tf_xTr, 1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_yTr))
        loss += lambda_loss * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))

        # Optimizer (Built into tensor flow, based on gradient descent)
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.01, batch * batch_size, nTr, 0.95, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

        # Predictions for the training, validation, and test data
        preds_Tr = tf.nn.softmax(model(tf_xTr_full, 0))
        preds_Te = tf.nn.softmax(model(tf_xTe, 0))

    # Feed data into the graph, run the model
    with tf.Session(graph=graph) as session:

        var_dict = {'w1': w1,
                    'b1': b1,
                    'w2': w2,
                    'b2': b2,
                    'w3': w3,
                    'b3': b3,
                    }
        saver = tf.train.Saver(var_dict)

        # Run model
        tf.initialize_all_variables().run()
        print('Graph initialized ...')
        t = time.time()
        for step in range(num_steps):
            offset = (step * batch_size) % (nTr - batch_size)
            batch_data = xTr[offset:(offset + batch_size), :]
            batch_labels = yTr[offset:(offset + batch_size), :]
            feed_dict = {tf_xTr: batch_data, tf_yTr: batch_labels}
            _ = session.run([optimizer], feed_dict=feed_dict)

            if step % 5000 == 0:
                l, preds_Train, preds_Test = session.run([loss, preds_Tr, preds_Te], feed_dict=feed_dict)
                # Find max and set to 1, else 0
                for i in range(nTr):
                    ind_Tr = np.argsort(preds_Train[i, :])[::-1][:5]
                    preds_Train[i, :] = 0
                    for j in range(1, 6):
                        preds_Train[i, ind_Tr[j - 1]] = j
                for i in range(nTe):
                    ind_Te = np.argsort(preds_Test[i, :])[::-1][:5]
                    preds_Test[i, :] = 0
                    for j in range(1, 6):
                        preds_Test[i, ind_Te[j - 1]] = j
                acc_Tr = accuracy(preds_Train, yTr)
                acc_Te = accuracy(preds_Test, yTe)

                print('Minibatch loss at step %d: %f' % (step, l))
                print('Training accuracy of top 5 probabilities: %s' % acc_Tr)
                print('Testing accuracy of top 5 probabilities: %s' % acc_Te)
                print('Time consumed: %d minutes' % ((time.time() - t) / 60.))
                saver.save(session, ckpt_dir + 'model.ckpt', global_step=step + 1)

            elif step % 500 == 0:
                print('Step %d complete ...' % step)

            # if step == 10000:
            #     break

    print('Training complete.')


if __name__ == '__main__':

    # Define batch size for SGD, and network architecture
    batch_size = 128
    num_nodes_layer1 = 512
    num_nodes_layer2 = 512
    board_height = 8
    board_width = 4
    label_height = 32
    label_width = 4

    # Extract training data into win_dict, loss_dict, and draw_dict
    trainset_file = 'checkers_library_full_v2.pickle'
    win_dict, loss_dict, draw_dict = rp.read_pickle(trainset_file)
    print('Finished loading data.')

    # Create numpy arrays xTr(nx8x4) and yTr(nx32x4), where n = number of training examples
    data_list = list()
    labels_list = list()
    for dictionary in [win_dict, loss_dict, draw_dict]:
        for key in dictionary:
            data_list.append(dictionary[key][0].as_matrix())
            labels_list.append(dictionary[key][1].as_matrix())
    data = np.array(data_list, dtype=int)
    labels = np.array(labels_list, dtype=int)

    # Randomize order since incoming data is structured into win, loss, draw
    n = len(data_list)
    assert n == len(labels_list)
    ind = np.arange(n)
    np.random.shuffle(ind)
    data, labels = data[ind, :], labels[ind, :]

    # Vectorize the inputs and labels
    data = data.reshape((-1, board_height * board_width)).astype(np.float32)
    labels = labels.reshape((-1, label_height * label_width)).astype(np.float32)

    # Split x, y into training, cross validation, and test sets
    test_split = 0.35
    nTe = int(test_split * n)
    nTr = n - nTe
    xTe, yTe = data[:nTe, :], labels[:nTe, :]
    xTr, yTr = data[nTe:, :], labels[nTe:, :]
    assert n == nTr + nTe
    del data, labels

    param_dir = 'parameters_v2/fc_512_no_reg/'

    deepnet(num_steps=150001,
            lambda_loss=0,
            dropout_L1=0,
            dropout_L2=0.1,
            ckpt_dir=param_dir)