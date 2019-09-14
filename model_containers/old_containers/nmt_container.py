from __future__ import print_function
import sys
import os
import rpc
import json

import tensorflow as tf
import numpy as np

import gnmt_model
import model_helper
import misc_utils as utils
import hparam_utils

GPU_MEM_FRAC = .95

NMT_TEXT_END = "</s>"
NUM_TRANSLATIONS_PER_INPUT = 1

# These hyperparameters are required for inference and are not specified
# by the provided set of JSON-formatted GNMT model hyperparameters

class NMTContainer(rpc.ModelContainerBase):

  def __init__(self, checkpoint_path, default_hparams_path, model_hparams_path, source_vocab_path, target_vocab_path):
    """
    Initializes the container

    Parameters
    ------------
    checkpoint_path : str
      The path to the GNMTModel checkpoint
    default_hparams_path : str
      The path to the set of default hyperparameters holding the same values as
      the flags specified in the `nmt.py` file
    model_hparams_path : str
      The path to the set of tuned GNMT hyperparameters
    source_vocab_path : str
      The path of the vocabulary associated with the source text (German)
    target_vocab_path : str
      The path of the vocabulary associated with the target text (English)
    """
    self.sess, self.nmt_model, self.infer_model, self.hparams = \
    self._load_model(checkpoint_path,
                     default_hparams_path,
                     model_hparams_path,
                     source_vocab_path,
                     target_vocab_path)


  def predict_bytes(self, inputs):
    """
    Parameters
    -------------
    inputs : [string]
      A list of strings of German text
    """
    inputs = [str(input_item.tobytes()) for input_item in inputs]

    infer_batch_size = len(inputs)
    self.sess.run(
        self.infer_model.iterator.initializer,
        feed_dict={
            self.infer_model.src_placeholder: inputs,
            self.infer_model.batch_size_placeholder: infer_batch_size
    })

    outputs = []

    nmt_outputs, _ = self.nmt_model.decode(self.sess)
    for output_id in range(infer_batch_size):
      for translation_index in range(NUM_TRANSLATIONS_PER_INPUT):
        output = self._get_translation(nmt_outputs[translation_index],
                                       output_id,
                                       tgt_eos=None,
                                       subword_option=self.hparams.subword_option)
        end_idx = output.find(NMT_TEXT_END)
        if end_idx >= 0:
            output = output[:end_idx]
        outputs.append(output)

    return outputs

  def _create_hparams(self, default_hparams_path, model_hparams_path, source_vocab_path, target_vocab_path):
    partial_hparams = tf.contrib.training.HParams()
    default_hparams_file = open(default_hparams_path, "rb")
    default_hparams = json.load(default_hparams_file)
    default_hparams_file.close()
    for param in default_hparams:
      partial_hparams.add_hparam(param, default_hparams[param])
    partial_hparams.set_hparam("num_gpus", 1)

    hparams = hparam_utils.load_hparams(model_hparams_path, partial_hparams)
    hparams = hparam_utils.extend_hparams(hparams, source_vocab_path, target_vocab_path)
    return hparams

  def _load_model(self,
                  checkpoint_path,
                  default_hparams_path,
                  model_hparams_path,
                  source_vocab_path,
                  target_vocab_path):
    hparams = self._create_hparams(default_hparams_path, model_hparams_path, source_vocab_path, target_vocab_path)

    model_creator = gnmt_model.GNMTModel
    infer_model = model_helper.create_infer_model(model_creator, hparams, scope=None)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRAC)
    sess = tf.Session(graph=infer_model.graph, config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    with infer_model.graph.as_default():
      nmt_model = model_helper.load_model(infer_model.model, checkpoint_path, sess, "infer")

    return sess, nmt_model, infer_model, hparams

  def _get_translation(self, nmt_outputs, sent_id, tgt_eos, subword_option):
    """Given batch decoding outputs, select a sentence and turn to text."""
    if tgt_eos:
      tgt_eos = tgt_eos.encode("utf-8")
    # Select a sentence
    output = nmt_outputs[sent_id, :].tolist()

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
      output = output[:output.index(tgt_eos)]

    if subword_option is None:
      translation = utils.format_text(output)
    elif subword_option == "bpe":  # BPE
      translation = utils.format_bpe_text(output)

    if subword_option == "spm":  # SPM
      translation = utils.format_spm_text(output)

    return translation


if __name__ == "__main__":
    print("Starting NMT Container")
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
        model_checkpoint_path = os.environ["CLIPPER_MODEL_CHECKPOINT_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_CHECKPOINT_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_default_hparams_path = os.environ["CLIPPER_DEFAULT_HPARAMS_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_DEFAULT_HPARAMS_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_hparams_path = os.environ["CLIPPER_MODEL_HPARAMS_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_HPARAMS_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_source_vocab_path = os.environ["CLIPPER_SOURCE_VOCAB_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_SOURCE_VOCAB_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_target_vocab_path = os.environ["CLIPPER_TARGET_VOCAB_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_TARGET_VOCAB_PATH environment variable must be set",
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

    input_type = "bytes"
    container = NMTContainer(model_checkpoint_path,
                             model_default_hparams_path,
                             model_hparams_path,
                             model_source_vocab_path,
                             model_target_vocab_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, model_name, model_version,
                      input_type)
