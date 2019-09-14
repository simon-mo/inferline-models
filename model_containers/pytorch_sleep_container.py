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

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class TorchContainer(rpc.ModelContainerBase):
    def __init__(self, model_arch):
        logger.info("Using PyTorch Alexnet")
        self.model = models.alexnet(pretrained=True)

        if torch.cuda.is_available():
            self.model.cuda()

        self.model.eval()
        self.height = 299
        self.width = 299

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.preprocess = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    def predict_floats(self, inputs):
        batch_size = 30
        num_batches = (len(inputs) // batch_size) + ((len(inputs) % batch_size) > 0)
        # for _ in range(num_batches):
        #     time.sleep(0.178)
        return np.random.random(len(inputs)).astype(str).tolist()


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

    model_arch = "alexnet"
    model = TorchContainer(model_arch)
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, port, model_name, model_version, input_type, start_time)
