from __future__ import print_function
import sys
import os
import rpc
import numpy as np
import gensim

class SimilarityModelContainer(rpc.ModelContainerBase):

    def __init__(self, model_path, dictionary_path):
        self.word_ids_dict = gensim.corpora.Dictionary.load_from_text(dictionary_path)
        self.model = gensim.similarities.docsim.SparseMatrixSimilarity.load(model_path, mmap='r')

    def predict_strings(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            A list of documents, represented as strings

        Returns
        ----------
        list 
            A list of document ids. The output at index `i`
            is the index of the document predicted to be most
            similar to the input document at index `i`
        """

        outputs = []
        for input_doc in inputs:
            doc_bow = self.word_ids_dict.doc2bow(input_doc.split())
            docsim_dist = self.model[doc_bow]
            best_doc_index = np.argmax(docsim_dist)
            outputs.append(str(best_doc_index))

        return outputs

if __name__ == "__main__":
    print("Starting Gensim Document Similarity Container")
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
    try:
        dict_path = os.environ["CLIPPER_DICT_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_DICT_PATH environment variable must be set",
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

    input_type = "strings"
    container = SimilarityModelContainer(model_path, dict_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version,
                      input_type)