# -*- coding: utf-8 -*-
from __future__ import division

import os
import math
import numpy as np
import scipy.interpolate
import tensorflow as tf

import utils as utils_mod
from utils import *


def test(config):
    # ==============================
    # Basic settings
    # ==============================
    test_number = 100000

    # Network parameters
    n_hidden_1 = 500
    n_hidden_2 = 250
    n_hidden_3 = 120

    n_input = 256
    n_output = len(config.pred_range)

    # ==============================
    # OFDM parameters
    # ==============================
    H_folder = config.Test_set_path
    SNRdb = config.SNR
    P = config.Pilots

    mu = 2
    K = 64
    CP = K // 4

    CP_flag = config.with_CP_flag
    Clipping_Flag = config.Clipping

    allCarriers = np.arange(K)

    if P > 0 and P < K:
        pilotCarriers = allCarriers[::K // P]
        pilotCarriers = pilotCarriers[:P]
    elif P >= K:
        pilotCarriers = allCarriers
    else:
        pilotCarriers = np.array([], dtype=int)

    dataCarriers = np.delete(allCarriers, pilotCarriers)

    if P >= K:
        pilotValue = np.ones(K, dtype=complex) * (3 + 3j)
    else:
        pilotValue = 3 + 3j

    payloadBits_per_OFDM = K * mu

    utils_mod.P = P
    utils_mod.SNRdb = SNRdb

    # ==============================
    # Reset TensorFlow graph
    # ==============================
    tf.reset_default_graph()

    # ==============================
    # TensorFlow placeholders
    # ==============================
    X = tf.placeholder(tf.float32, [None, n_input], name="X")
    Y = tf.placeholder(tf.float32, [None, n_output], name="Y")

    # ==============================
    # DNN model
    # ==============================
    def encoder(x):
        weights = {
            'encoder_h1': tf.Variable(
                tf.truncated_normal([n_input, n_hidden_1], stddev=0.1)
            ),
            'encoder_h2': tf.Variable(
                tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)
            ),
            'encoder_h3': tf.Variable(
                tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)
            ),
            'encoder_h4': tf.Variable(
                tf.truncated_normal([n_hidden_3, n_output], stddev=0.1)
            ),
        }

        biases = {
            'encoder_b1': tf.Variable(
                tf.truncated_normal([n_hidden_1], stddev=0.1)
            ),
            'encoder_b2': tf.Variable(
                tf.truncated_normal([n_hidden_2], stddev=0.1)
            ),
            'encoder_b3': tf.Variable(
                tf.truncated_normal([n_hidden_3], stddev=0.1)
            ),
            'encoder_b4': tf.Variable(
                tf.truncated_normal([n_output], stddev=0.1)
            ),
        }

        layer_1 = tf.nn.relu(
            tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1'])
        )
        layer_2 = tf.nn.relu(
            tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2'])
        )
        layer_3 = tf.nn.relu(
            tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3'])
        )
        layer_4 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4'])
        )

        return layer_4

    y_pred = encoder(X)
    y_true = Y

    mean_error_tensor = tf.reduce_mean(tf.abs(y_true - y_pred))

    pred_bits = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    true_bits = tf.cast(tf.greater_equal(y_true, 0.5), tf.float32)

    bit_errors = tf.not_equal(pred_bits, true_bits)
    BER_tensor = tf.reduce_mean(tf.cast(bit_errors, tf.float32))

    # ==============================
    # Load channel response set
    # ==============================
    test_idx_low = 1
    test_idx_high = 80

    channel_response_set_test = []

    for test_idx in range(test_idx_low, test_idx_high):
        H_file = os.path.join(H_folder, str(test_idx) + '.txt')

        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]

                half_len = int(len(numbers_float) / 2)
                real_part = np.asarray(numbers_float[0:half_len])
                imag_part = np.asarray(numbers_float[half_len:len(numbers_float)])

                h_response = real_part + 1j * imag_part
                channel_response_set_test.append(h_response)

    print("length of testing channel response", len(channel_response_set_test))

    # ==============================
    # TensorFlow session
    # ==============================
    saver = tf.train.Saver()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())

        # ==============================
        # Restore model
        # ==============================
        ckpt_dir = os.path.join(config.Model_path, "SNR_{}".format(config.SNR))
        saving_name = tf.train.latest_checkpoint(ckpt_dir)

        if saving_name is None:
            raise ValueError("No checkpoint found in {}".format(ckpt_dir))

        print("Restoring model from:", saving_name)
        saver.restore(sess, saving_name)

        # ==============================
        # Generate testing data
        # ==============================
        input_samples_test = []
        input_labels_test = []

        for i in range(test_number):
            bits = np.random.binomial(
                n=1,
                p=0.5,
                size=(payloadBits_per_OFDM,)
            )

            channel_response = channel_response_set_test[
                np.random.randint(0, len(channel_response_set_test))
            ]

            signal_output, para = utils_mod.ofdm_simulate(
                bits,
                channel_response,
                SNRdb,
                mu,
                CP_flag,
                K,
                P,
                CP,
                pilotValue,
                pilotCarriers,
                dataCarriers,
                Clipping_Flag
            )

            input_labels_test.append(bits[config.pred_range])
            input_samples_test.append(signal_output)

        batch_x = np.asarray(input_samples_test)
        batch_y = np.asarray(input_labels_test)

        # ==============================
        # Evaluate BER
        # ==============================
        mean_error_value, BER_value = sess.run(
            [mean_error_tensor, BER_tensor],
            feed_dict={
                X: batch_x,
                Y: batch_y
            }
        )

        print(
            "OFDM Detection QAM output number is",
            n_output,
            "SNR =",
            SNRdb,
            "Num Pilot",
            P,
            "mean error =",
            mean_error_value,
            "BER =",
            BER_value
        )

    return BER_value