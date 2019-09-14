from __future__ import print_function
import sys
import os
import rpc
import numpy as np

import cv2

NUM_SIFT_FEATURES = 20

class SIFTFeaturizationContainer(rpc.ModelContainerBase):

    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=NUM_SIFT_FEATURES)

    def predict_floats(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            A list of images, each of which is represented
            as a numpy array of floats
        """
        inputs = [input_item.reshape((299,299,3)).astype(np.uint8) for input_item in inputs]
        return [self._get_keypoints(input_img) for input_img in inputs]

    def _get_keypoints(self, img):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keypoints, features = self.sift.detectAndCompute(grayscale_img, None)
        return np.array(features[:NUM_SIFT_FEATURES], dtype=np.int32)

if __name__ == "__main__":
    print("Starting OpenCV SIFT Featurization Container")
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

    input_type = "floats"
    container = SIFTFeaturizationContainer()
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version,
                      input_type)
