from __future__ import print_function
import sys
import os
import rpc
import pickle

from sklearn.svm import LinearSVC

class OpenCvSvmContainer(rpc.ModelContainerBase):

    def __init__(self, model_path):
        model_file = open(model_path, "rb")
        self.model = pickle.load(model_file)
        model_file.close()

    def predict_ints(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            A list of SIFT feature vectors, each
            represented as numpy array of data type `np.int32`
        """
        outputs = self.model.predict(inputs)
        print(outputs)
        print(type(outputs))
        return self.model.predict(inputs)

if __name__ == "__main__":
    print("Starting OpenCV SVM Container")
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
        model_path = os.environ["CLIPPER_MODEL_PATH"]
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

    input_type = "ints"
    container = OpenCvSvmContainer(model_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version,
                      input_type)
