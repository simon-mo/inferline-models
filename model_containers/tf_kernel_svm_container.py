from __future__ import print_function
import sys
import os
import rpc
import numpy as np
import tensorflow as tf
from datetime import datetime

# ResNet feature vectors are of size 2048
INPUT_VECTOR_SIZE = 2048

class TFKernelSvmContainer(rpc.ModelContainerBase):

    def __init__(self, kernel_size=2000, gpu_mem_frac=.95):
        self.kernel_data = self._generate_kernel_data(kernel_size)
        self.weights = self._generate_weights(kernel_size)
        self.labels = self._generate_labels(kernel_size)
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
            self.t_kernel : self.kernel_data,
            self.t_weights : self.weights,
            self.t_labels : self.labels,
            self.t_bias : self.bias,
            self.t_inputs : inputs
        }

        outputs = self.sess.run(self.t_outputs, feed_dict=feed_dict)
        outputs = outputs.flatten()

        return [np.array(item, dtype=np.float32) for item in outputs]

    def _create_prediction_graph(self):
        with tf.device("/gpu:0"):
            self.t_kernel = tf.placeholder(tf.float32, [None, INPUT_VECTOR_SIZE])
            self.t_inputs = tf.placeholder(tf.float32, [None, INPUT_VECTOR_SIZE])
            self.t_weights = tf.placeholder(tf.float32, [None, 1])
            self.t_labels = tf.placeholder(tf.float32, [None, 1])
            self.t_bias = tf.placeholder(tf.float32)
            gamma = tf.constant(-50.0)

            # Taken from https://github.com/nfmcclure/tensorflow_cookbook
            rA = tf.reshape(tf.reduce_sum(tf.square(self.t_kernel), 1),[-1,1])
            rB = tf.reshape(tf.reduce_sum(tf.square(self.t_inputs), 1),[-1,1])
            pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2.0, tf.matmul(self.t_kernel, tf.transpose(self.t_inputs)))), tf.transpose(rB))
            pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

            t_preds = tf.matmul(tf.multiply(tf.transpose(tf.multiply(self.t_labels, self.t_weights)), self.t_bias), pred_kernel)
            self.t_outputs = tf.sign(t_preds - tf.reduce_mean(t_preds))

    def _generate_bias(self):
        return np.random.uniform(-1,1) * 100

    def _generate_weights(self, training_data_size):
        return np.random.uniform(-1,1, size=(training_data_size, 1))

    def _generate_labels(self, training_data_size):
        return np.array(np.random.choice([-1,1], size=(training_data_size, 1)), dtype=np.float32)

    def _generate_kernel_data(self, kernel_size):
        return np.random.rand(kernel_size, INPUT_VECTOR_SIZE) * 10

if __name__ == "__main__":
    start_time = datetime.now()
    print("Starting Tensorflow Kernel SVM Container")
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

    kernel_size = 2000
    if "CLIPPER_KERNEL_SIZE" in os.environ:
        kernel_size = int(os.environ["CLIPPER_KERNEL_SIZE"])
    else:
        print("Using default kernel size of {ks}".format(ks=kernel_size))

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
    container = TFKernelSvmContainer(kernel_size, gpu_mem_frac)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, model_name, model_version, input_type, start_time)
