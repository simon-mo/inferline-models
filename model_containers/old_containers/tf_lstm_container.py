from __future__ import print_function
import sys
import os
import rpc
import json
import re

import tensorflow as tf
import numpy as np

from tf_lstm_vocab import Vocabulary

GPU_MEM_FRAC = .95

# Sentiment is either positive or negative
NUM_CLASSES = 2
MAX_SEQ_LENGTH = 100
NUM_DIMENSIONS = 300
LSTM_UNITS = 64

RE_SPECIAL_CHARS = re.compile("[^A-Za-z0-9 ]+")

"""
Adapted from https://github.com/adeshpande3/LSTM-Sentiment-Analysis
"""
class TfLstmContainer(rpc.ModelContainerBase):

    def __init__(self, vocab_dir_path):
        self.vocabulary = Vocabulary(vocab_dir_path)
        self.sess, self.input_data, self.sentiment_scores = self._create_model_graph()

    def predict_bytes(self, inputs):
        """
        Parameters
        ------------
        inputs : [str]
            A list of string inputs in one of 64 languages
        """
        inputs = [str(input_item.tobytes()) for input_item in inputs]

        inputs_matrix = self._get_inputs_matrix(inputs)
        feed_dict = {
            self.input_data : inputs_matrix
        }

        sentiment_scores = self.sess.run(self.sentiment_scores, feed_dict=feed_dict)
        outputs = []
        for pos_score, neg_score in sentiment_scores:
            if pos_score >= neg_score:
                outputs.append(1)
            else:
                outputs.append(0)

        return [np.array(output, dtype=np.float32) for output in outputs]

    def _get_inputs_matrix(self, inputs):
        """
        Converts a batch of string inputs into a matrix such that
        the `n`th row is the vector representation of the `n`th
        input
        """

        inputs_matrix = np.zeros([len(inputs), MAX_SEQ_LENGTH], dtype='int32')
        cleaned_inputs = [self._clean_input(input_str) for input_str in inputs]
        for batch_idx in range(len(inputs)):
            split_input = cleaned_inputs[batch_idx].split()
            for word_idx, word in enumerate(split_input):
                inputs_matrix[batch_idx, word_idx] = self.vocabulary.get_word_idx(word)

        return inputs_matrix

    def _clean_input(self, input_str):
        # Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
        input_str = input_str.lower().replace("<br />", " ")
        return re.sub(RE_SPECIAL_CHARS, "", input_str)

    def _create_model_graph(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRAC)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        with tf.device("/gpu:0"):
            input_data = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH])

            data = tf.nn.embedding_lookup(self.vocabulary.get_word_vecs(), input_data)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_UNITS)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.25)
            value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

            weight = tf.Variable(tf.truncated_normal([LSTM_UNITS, NUM_CLASSES]))
            bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
            value = tf.transpose(value, [1, 0, 2])
            last = tf.gather(value, int(value.get_shape()[0]) - 1)

            sentiment_scores = tf.matmul(last, weight) + bias

        sess.run(tf.global_variables_initializer())

        return sess, input_data, sentiment_scores

if __name__ == "__main__":
    print("Starting TF Lstm Container!")
    try:
        model_name = os.environ["CLIPPER_MODEL_NAME"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_NAME environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_version = os.environ["CLIPPER_MODEL_VERSION"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_VERSION environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_vocab_path = os.environ["CLIPPER_MODEL_VOCAB_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_VOCAB_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)

    ip = "127.0.0.1"
    if "CLIPPER_IP" in os.environ:
        ip = os.environ["CLIPPER_IP"]
    else:
        print("Connecting to Clipper on localhost")

    print("CLIPPER IP: {}".format(ip))

    port = 7000
    if "CLIPPER_PORT" in os.environ:
        port = int(os.environ["CLIPPER_PORT"])
    else:
        print("Connecting to Clipper with default port: 7000")

    input_type = "bytes"
    container = TfLstmContainer(model_vocab_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, model_name, model_version,
                      input_type)
