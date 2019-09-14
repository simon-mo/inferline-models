from __future__ import print_function, absolute_import, division
import rpc
import os
import sys
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image, ImageEnhance
import logging
from datetime import datetime

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class TorchPreprocessContainer(rpc.ModelContainerBase):
    def __init__(self):
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

        sample = np.random.random(299*299*3)
        xx = self.predict_doubles([sample])
        print(type(xx[0]))
        print(xx[0].shape)
        print(xx)

    def predict_doubles(self, inputs):
        reshaped_inputs = []
        for i in inputs:
            x = i.reshape(self.height, self.width, 3)
            img = Image.fromarray(x, mode="RGB").resize((400, 400))
            sharpener = ImageEnhance.Sharpness(img)
            img = sharpener.enhance(1.8)
            brightener = ImageEnhance.Brightness(img)
            img = brightener.enhance(1.5)
            contraster = ImageEnhance.Contrast(img)
            img = contraster.enhance(1.5)
            reshaped_inputs.append(self.preprocess(img))
        return [np.array(r.numpy().flatten(), dtype=np.float32) for r in reshaped_inputs]
        # start = datetime.now()
        # input_arrs = []
        # for t in inputs:
        #     i = t.reshape(self.height, self.width, 3)
        #     input_arrs.append(i)
        # pred_classes = self._predict_raw(input_arrs)
        # if pred_classes.shape == ():
        #     outputs = [str(pred_classes)]
        # else:
        #     outputs = [str(l) for l in pred_classes]
        # end = datetime.now()
        # # logger.info("BATCH TOOK %f seconds" % (end - start).total_seconds())
        # return outputs

    def predict_floats(self, inputs):
        return self.predict_doubles(inputs)

    def bench(self, num_iters, batch_size):
        lats = []
        for i in range(num_iters):
            inputs = [np.random.random(299*299*3) for _ in range(batch_size)]
            start = datetime.now()
            self.predict_doubles(inputs)
            end = datetime.now()
            lats.append((end-start).total_seconds())
        print("Mean: {mean:.3f} ms, P99: {p99:.3f} ms".format(
            mean=1000.0*np.mean(lats), p99=1000.0*np.percentile(lats, 99)))



    # def predict_bytes(self, inputs):
    #     start = datetime.now()
    #     input_arrs = []
    #     for byte_arr in inputs:
    #         t = np.frombuffer(byte_arr, dtype=np.float32)
    #         i = t.reshape(self.height, self.width, 3)
    #         input_arrs.append(i)
    #     pred_classes = self._predict_raw(input_arrs)
    #     outputs = [str(l) for l in pred_classes]
    #     # logger.debug("Outputs: {}".format(outputs))
    #     end = datetime.now()
    #     # logger.info("BATCH TOOK %f seconds" % (end - start).total_seconds())
    #     return outputs

    # def _predict_raw(self, input_arrs):
    #     inputs = []
    #     for i in input_arrs:
    #         img = Image.fromarray(i, mode="RGB")
    #         inputs.append(self.preprocess(img))
    #     input_batch = Variable(torch.stack(inputs, dim=0))
    #     if torch.cuda.is_available():
    #         input_batch = input_batch.cuda()
    #     logits = self.model(input_batch)
    #     maxes, arg_maxes = torch.max(logits, dim=1)
    #     pred_classes = arg_maxes.squeeze().data.cpu().numpy()
    #     return pred_classes


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

    # mpath = os.environ["CLIPPER_MODEL_PATH"]
    # with open(mpath, "rb") as f:
    #     model_arch = f.read().strip()
    model = TorchPreprocessContainer()
    # model.bench(100, 4)
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, model_name, model_version, input_type, start_time)
