from __future__ import print_function
import sys
import os
import rpc
import json

import tensorflow as tf
import numpy as np
import cnn
import util
from datetime import datetime

GPU_MEM_FRAC = .95

"""
Adapted from https://github.com/may-/cnn-ld-tf
"""
class LangDetectContainer(rpc.ModelContainerBase):

    def __init__(self, config_path, checkpoint_path, vocab_path):
        self.sess, self.inputs_tensor, self.scores_tensor = self._load_model(config_path, checkpoint_path)
        self.vocab = self._load_vocab(vocab_path)

    def predict_bytes(self, inputs):
        """
        Parameters
        ------------
        inputs : [str]
            A list of string inputs in one of 64 languages
        """
        inputs = [str(input_item.tobytes()) for input_item in inputs]

        ids_inputs = np.array([self.vocab.text2id(input_text) for input_text in inputs])

        feed_dict = {
            self.inputs_tensor : ids_inputs
        }
        all_scores = self.sess.run(self.scores_tensor, feed_dict=feed_dict)

        outputs = []
        for score_dist in all_scores:
            parsed_dist = [float(str(i)) for i in score_dist]
            pred_class = self.vocab.class_names[int(np.argmax(parsed_dist))]
            outputs.append(str(pred_class.replace("#", "")))

        return outputs

    def _load_model(self, config_path, checkpoint_path):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRAC)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        config = self._load_config(config_path)

        with tf.device("/gpu:0"):
            with tf.variable_scope('cnn'):
                model = cnn.Model(config, is_train=False)

            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            inputs_tensor = model.inputs
            scores_tensor = model.scores

        return sess, inputs_tensor, scores_tensor

    def _load_config(self, config_path):
        config = util.load_from_dump(config_path)
        return config

    def _load_vocab(self, vocab_path):
        return util.VocabLoader(vocab_path)

if __name__ == "__main__":
    start_time = datetime.now()
    print("Starting TF Language Detection Container!")
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
        model_config_path = os.environ["CLIPPER_MODEL_CONFIG_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_CONFIG_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_checkpoint_path = os.environ["CLIPPER_MODEL_CHECKPOINT_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_CHECKPOINT_PATH environment variable must be set",
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
    container = LangDetectContainer(model_config_path,
                                    model_checkpoint_path,
                                    model_vocab_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, model_name, model_version, input_type, start_time)
