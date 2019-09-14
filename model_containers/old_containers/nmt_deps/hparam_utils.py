import sys
import os
import tensorflow as tf
import misc_utils as utils
import vocab_utils

def extend_hparams(hparams, source_vocab_path, target_vocab_path):
  """
  Extends the set of hyperparameters
  """
  if hparams.encoder_type == "bi" and hparams.num_layers % 2 != 0:
    raise ValueError("For bi, num_layers %d should be even" %
                     hparams.num_layers)
  if (hparams.attention_architecture in ["gnmt"] and
      hparams.num_layers < 2):
    raise ValueError("For gnmt attention architecture, "
                     "num_layers %d should be >= 2" % hparams.num_layers)

  if hparams.subword_option not in [None, "spm", "bpe"]:
    raise ValueError("subword option must be either None, spm, or bpe")
  if hparams.bpe_delimiter is not None and hparams.bpe_delimiter != "@@":
    raise ValueError("BPE delimiter value must be '@@' %s",
                     hparams.bpe_delimiter)
  if hparams.bpe_delimiter == "@@":
    # if bpe_delimiter is set, subword_option will automatically set to bpe
    if hparams.subword_option == "spm":
      raise ValueError("Unable to set the subword option to spm "
                       "if bpe delimiter is set")
    else:
      hparams.subword_option = "bpe"

  # Flags
  utils.print_out("# hparams:")
  utils.print_out("  train_prefix=%s" % hparams.train_prefix)
  utils.print_out("  dev_prefix=%s" % hparams.dev_prefix)
  utils.print_out("  test_prefix=%s" % hparams.test_prefix)
  utils.print_out("  out_dir=%s" % hparams.out_dir)

  # Set num_residual_layers
  if hparams.residual and hparams.num_layers > 1:
    if hparams.encoder_type == "gnmt":
      # The first unidirectional layer (after the bi-directional layer) in
      # the GNMT encoder can't have residual connection due to the input is
      # the concatenation of fw_cell and bw_cell's outputs.
      num_residual_layers = hparams.num_layers - 2
    else:
      num_residual_layers = hparams.num_layers - 1
  else:
    num_residual_layers = 0
  hparams.add_hparam("num_residual_layers", num_residual_layers)

  src_vocab_file = source_vocab_path
  tgt_vocab_file = target_vocab_path

  # Source vocab
  src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
      src_vocab_file,
      hparams.out_dir,
      check_special_token=hparams.check_special_token,
      sos=hparams.sos,
      eos=hparams.eos,
      unk=vocab_utils.UNK)

  # Target vocab
  if hparams.share_vocab:
    utils.print_out("  using source vocab for target")
    tgt_vocab_file = src_vocab_file
    tgt_vocab_size = src_vocab_size
  else:
    tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
        tgt_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)
  hparams.add_hparam("src_vocab_size", src_vocab_size)
  hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
  hparams.add_hparam("src_vocab_file", src_vocab_file)
  hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

  return hparams


def load_hparams(hparams_path, default_hparams):
  """
  Loads hyperparameters from the specified path
  """
  return utils.maybe_parse_standard_hparams(default_hparams, hparams_path)