from __future__ import print_function
import sys
import os
import rpc

import base64
import tensorflow as tf
import numpy as np

sys.path.append("/tf_models/research/slim")

from nets import inception_v3
from preprocessing import inception_preprocessing
from datasets import imagenet

image_size = inception_v3.inception_v3.default_image_size
slim = tf.contrib.slim

class InceptionClassificationContainer(rpc.ModelContainerBase):

    def __init__(self, checkpoint_path):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.device("/gpu:0"):
            self.inputs = tf.placeholder(tf.float32, (None, image_size, image_size, 3))
            preprocessed_images = tf.map_fn(lambda input_img : inception_preprocessing.preprocess_image(input_img, image_size, image_size, is_training=False), self.inputs)

            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits, _ = inception_v3.inception_v3(preprocessed_images, num_classes=1001, is_training=False)
                self.all_probabilities = tf.nn.softmax(logits)
                init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables("InceptionV3"))

                init_fn(self.sess)

    def predict_floats(self, inputs):
        """
        Parameters
        -------------
        inputs : np.ndarray
            An image, represented as a flattened 299 x 299 x 3 
            numpy array of floats
        """
        reshaped_inputs = [input_item.reshape((299, 299, 3)) for input_item in inputs]
        all_probabilities = self.sess.run([self.all_probabilities], feed_dict={self.inputs: reshaped_inputs})

        outputs = []
        for input_probabilities in all_probabilities[0]:
            sorted_inds = [i[0] for i in sorted(
                enumerate(-input_probabilities), key=lambda x:x[1])]
            outputs.append(str(sorted_inds[0]))

        return outputs

if __name__ == "__main__":
    print("Starting Inception Classification Container")
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
        model_checkpoint_path = os.environ["CLIPPER_MODEL_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_PATH environment variable must be set",
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

    input_type = "floats"
    container = InceptionClassificationContainer(model_checkpoint_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version,
                      input_type)
