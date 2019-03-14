import gensim
import os
import numpy as np
from tqdm import tqdm
import time
from . import constant

from ..utils import utils


class Word2Vec(object):
    """ A Word2Vec is defined as Word embedding model containing of
            - Word2Vec (implemented by gensim) :
                input format : list of string (e.g. ["ฉัน", "กิน", "ข้าว"])
                output format : list of vectors has length which equals the number of words (e.g. [v1, v2, v3])

                Moreover, Word2Vec is trained by data which are BEST_data, free BEST_data, Pantip posts, TED_thai.txt
                by input format such as [s1, s2, s3]
                which s1, s2, s3 are defined to be sentence (list of words). For example, s1 can be ["ฉัน", "เดิน"].

                To handle unseen word, the model uses low frequent word vectors instead.
    """

    def __init__(self, model_path=None, new_model=False):
        self.new_model = new_model
        self.model_path = model_path

        self.model = gensim.models.Word2Vec(
            size=constant.EMBEDDING_SIZE, sg=constant.CBOW_SG_FLAG, iter=5)

        if not new_model:
            if model_path is not None:
                self.model = gensim.models.Word2Vec.load(model_path)
            else:
                self.model = gensim.models.Word2Vec.load(
                    constant.DEFULT_MODEL_PATH)
        else:
            if model_path is not None:
                raise ValueError('model_path must be none for new model')

    def train(self, sentences):
        """
            Parameter :
                pre_process is to transform any raw string ("ฉันกินข้าว") into one training sentence (["ฉัน", "กิน", "ข้าว"])
                update is to build initial vocab for the model or update it

            Word2Vec model is implemented by gensim library with default parameters
        """

        print("Start finding low frequency word to be UNK")
        # find all low frequency words, then change it to be "UNK"
        low_freq_word = utils.find_low_freq_words(sentences, threshold=5)
        for idx in tqdm(range(len(sentences))):
            time.sleep(0.1)
            lst = sentences[idx]
            for ind, item in enumerate(lst):
                lst[ind] = low_freq_word.get(item, item)

        for sentence in sentences:
            if constant.UNKNOWN_WORD in sentence:
                print("CHECKPOINT: There is UNK word in sentences!")
                break

        # If new_model (empty) then the model should first build the vocab, update=False
        print("Start training model.")

        if not self.new_model:
            self.model.build_vocab(sentences, update=True)
            self.model.train(
                sentences, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        else:
            self.model.build_vocab(sentences, update=False)
            self.model.train(
                sentences, total_examples=self.model.corpus_count, epochs=self.model.epochs)

        self.new_model = False

    def predict(self, word):

        try:
            self.model[word]
        except KeyError:
            #             print("{} not found! the vector will be unknown word instead.".format(word))
            return self.model[constant.UNKNOWN_WORD]
        else:
            return self.model[word]

    def evaluate(self, pairs):
        # This expects input to be list of pairs of word, ex. [("กิน", "รับประทาน"), ("นอน", "หลับ")]
        # and will return the list of those pairs with similarity
        ret = []
        for (first_word, second_word) in pairs:
            # check if the word is in vocab
            try:
                self.model[first_word]
            except KeyError:
                print("The word {} is not in vocabulary !".format(first_word))
            else:
                try:
                    self.model[second_word]
                except KeyError:
                    print("The word {} is not in vocabulary !".format(second_word))
                else:
                    ret.append(
                        (first_word, second_word, self.model.wv.similarity(first_word, second_word)))
        return ret

    def save(self, filepath):
        """ Save the gensim model to a .bin file """

        if os.path.exists(filepath):
            raise ValueError("File " + filepath + " already exists!")
        self.model.save(filepath)
