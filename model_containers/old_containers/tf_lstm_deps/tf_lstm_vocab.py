import sys
import os
import numpy as np

RELATIVE_PATH_WORD_LIST = "word_list.npz"
RELATIVE_PATH_WORD_VECTORS = "word_vecs.npz"

# scalar for unknown words
WORD_VAL_UNKNOWN = 399999

class Vocabulary:

    def __init__(self, vocab_dir_path):
        word_list_path = os.path.join(vocab_dir_path, RELATIVE_PATH_WORD_LIST)
        word_vectors_path = os.path.join(vocab_dir_path, RELATIVE_PATH_WORD_VECTORS)

        assert os.path.exists(word_list_path)
        assert os.path.exists(word_vectors_path)

        word_list_file = open(word_list_path, "rb")
        word_list = [word.decode('UTF-8') for word in np.load(word_list_file)]
        word_list_file.close()
        self.word_dict = {}
        for i in range(len(word_list)):
            self.word_dict[word_list[i]] = i

        word_vectors_file = open(word_vectors_path, "rb")
        self.word_vectors = np.load(word_vectors_file)
        word_vectors_file.close()

    def get_word_idx(self, word):
        try:
            return self.word_dict[word]
        except KeyError:
            return WORD_VAL_UNKNOWN

    def get_word_vecs(self):
        return self.word_vectors


