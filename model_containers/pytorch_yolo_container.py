from __future__ import print_function, absolute_import, division
import rpc
import os
import sys
import numpy as np
import torch
from torchvision import models, transforms
# from torch.autograd import Variable
# from PIL import Image
import logging
from datetime import datetime
import time

from darknet import Darknet

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

CONFIG = {
     'cfgfile': 'cfg/yolov3.cfg',
     'weightsfile': 'yolov3.weights',
     'inp_dim': 224, # dim needs to be multiple of 32
}

class TorchContainer(rpc.ModelContainerBase):
    def __init__(self, config):
        model = Darknet(config['cfgfile'])
        model.load_weights(config['weightsfile'])
        model.net_info["height"] = str(config['inp_dim'])
        self.model = model.cuda().eval()

        self.inp_dim = int(model.net_info["height"])


    def predict_floats(self, inputs):
        reshaped_arr = np.array(inputs).astype(np.float32).reshape(len(inputs), 3, self.inp_dim, self.inp_dim)
        inp = torch.tensor(reshaped_arr).cuda()
        with torch.no_grad():
            out = self.model(inp, True)
        return ['json_serialized_output_placeholder' for _ in range(len(inputs))]


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

    port = 7000
    if "CLIPPER_PORT" in os.environ:
        port = int(os.environ["CLIPPER_PORT"])
    else:
        print("Connecting to Clipper with default port: 7000")

    input_type = "floats"
    if "CLIPPER_INPUT_TYPE" in os.environ:
        input_type = os.environ["CLIPPER_INPUT_TYPE"]

    model = TorchContainer(CONFIG)
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, model_name, model_version, input_type, start_time)
