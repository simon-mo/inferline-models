from __future__ import print_function
import sys
import os
import rpc
import numpy as np
import tensorflow as tf
from datetime import datetime

# Inception feature vectors are of size 2048
INPUT_VECTOR_SIZE = 2048

class TFLogRegContainer(rpc.ModelContainerBase):

    def __init__(self, gpu_mem_frac=.95):
        self.weights = self._generate_weights()
        self.bias = self._generate_bias()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        self._create_prediction_graph()


    def predict_floats(self, inputs):
        """
        Parameters
        --------------
        inputs : [np.ndarray]
            A list of float vectors of length 2048,
            represented as numpy arrays
        """

        feed_dict = {
            self.t_weights : self.weights,
            self.t_bias : self.bias,
            self.t_inputs : inputs
        }

        outputs = self.sess.run(self.t_outputs, feed_dict=feed_dict)
        outputs = outputs.flatten()

        return [np.array(item, dtype=np.float32) for item in outputs]

    def _create_prediction_graph(self):
        with tf.device("/gpu:0"):
            self.t_inputs = tf.placeholder(tf.float32, [None, INPUT_VECTOR_SIZE])
            self.t_weights = tf.placeholder(tf.float32, [INPUT_VECTOR_SIZE, 1])
            self.t_bias = tf.placeholder(tf.float32)

            t_apply_weights = tf.reduce_sum(tf.multiply(self.t_weights, tf.transpose(self.t_inputs)), axis=0)
            t_sig_input = t_apply_weights + self.t_bias

            self.t_outputs = tf.sigmoid(t_sig_input)

    def _generate_bias(self):
        return np.random.uniform(-1,1) * 100

    def _generate_weights(self):
        return np.random.uniform(-1,1, size=(INPUT_VECTOR_SIZE,1))

if __name__ == "__main__":
    start_time = datetime.now()
    print("Starting Tensorflow Logistic Regression Container")
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

    gpu_mem_frac = .9
    if "CLIPPER_GPU_MEM_FRAC" in os.environ:
        gpu_mem_frac = float(os.environ["CLIPPER_GPU_MEM_FRAC"])
    else:
        print("Using all available GPU memory")

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

    input_type = "floats"
    container = TFLogRegContainer(gpu_mem_frac)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, model_name, model_version, input_type, start_time)
