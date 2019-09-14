from __future__ import print_function
import sys
import os
import rpc

import numpy as np
import tensorflow as tf
from datetime import datetime

class TfResNetContainer(rpc.ModelContainerBase):

    def __init__(self, model_graph_path, model_ckpt_path):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._load_model(model_graph_path, model_ckpt_path)

    def predict_floats(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            A list of 3-channel, 224 x 224 images, each represented
            as a flattened numpy array
        """

        reshaped_inputs = [input_item.reshape(224,224,3) for input_item in inputs]
        all_img_features = self._get_image_features(reshaped_inputs)
        return [np.array(item, dtype=np.float32) for item in all_img_features]

    def _get_image_features(self, images):
        feed_dict = { self.t_images : images }
        features = self.sess.run(self.t_features, feed_dict=feed_dict)
        return features

    def _load_model(self, model_graph_path, model_ckpt_path):
        with tf.device("/gpu:0"):
            saver = tf.train.import_meta_graph(model_graph_path, clear_devices=True)
            saver.restore(self.sess, model_ckpt_path)
            self.t_images = tf.get_default_graph().get_tensor_by_name('images:0')
            self.t_features = tf.get_default_graph().get_tensor_by_name('avg_pool:0')


if __name__ == "__main__":
    start_time = datetime.now()
    print("Starting Tensorflow ResNet152 Container")
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
        model_graph_path = os.environ["CLIPPER_MODEL_GRAPH_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_checkpoint_path = os.environ["CLIPPER_MODEL_CHECKPOINT_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_CHECKPOINT_PATH environment variable must be set",
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
    container = TfResNetContainer(model_graph_path, model_checkpoint_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, model_name, model_version, input_type, start_time)
