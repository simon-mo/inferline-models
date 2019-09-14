from __future__ import print_function, absolute_import, division
import rpc
import os
import sys
import numpy as np
import logging
import json
from datetime import datetime

from openalpr import Alpr

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class ALPRContainer(rpc.ModelContainerBase):
    def __init__(self):
        self.alpr = Alpr(
            country='us',
            config_file='/etc/openalpr/openalpr.conf',
            runtime_dir='/usr/share/openalpr/runtime_data'
            )

    def predict_floats(self, inputs):
        int_inputs = (np.array(inputs) * 255).astype(np.uint8)
        reshaped_input = [arr.reshape(224, 224, 3) for arr in int_inputs]
        results = [self.alpr.recognize_ndarray(inp) for inp in reshaped_input]
        serialized = [json.dumps(result) for result in results]
        return serialized


if __name__ == "__main__":
    start_time = datetime.now()
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

    input_type = "floats"
    if "CLIPPER_INPUT_TYPE" in os.environ:
        input_type = os.environ["CLIPPER_INPUT_TYPE"]

    model = ALPRContainer()
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, model_name, model_version, input_type, start_time)
