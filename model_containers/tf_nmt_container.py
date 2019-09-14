from __future__ import print_function
import sys
import os
import rpc

import numpy as np

import tensorflow as tf

from nmt.model import Model
from nmt.utils.misc_utils import load_hparams, get_config_proto
from nmt.inference import inference, single_worker_inference
from nmt.model_helper import create_infer_model, load_model
from datetime import datetime


class TfNMTContainer(rpc.ModelContainerBase):

    def __init__(self):
        hparams = load_hparams('/tmp/nmt_model')
        ckpt = tf.train.latest_checkpoint('/tmp/nmt_model')
        self.model = create_infer_model(Model, hparams)
        self.sess = tf.Session(graph=self.model.graph, config=get_config_proto())
        with self.model.graph.as_default():
            self.loaded_infer_model = load_model(
                self.model.model, ckpt, self.sess, "infer")

    def predict_bytes(self, inputs):
        """
        Parameters
        ----------
        inputs : list
        A list of string to translate
        """
        inputs = [i.tostring() for i in inputs]
        self.sess.run(
        self.model.iterator.initializer,
        feed_dict={
            self.model.src_placeholder: inputs,
            self.model.batch_size_placeholder: len(inputs)
        })
        result, _ = self.loaded_infer_model.decode(self.sess)
        result_str = [' '.join(r) for r in result]
        return result_str


if __name__ == "__main__":
    start_time = datetime.now()
    print("Starting Tensorflow NMT Container")
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
    container = TfNMTContainer()
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, model_name, model_version, input_type, start_time)
