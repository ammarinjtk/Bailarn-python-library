from . import constant

import gensim
import logging
import itertools
import os
import deepcut
import tensorflow as tf

from .matric import custom_matric


class KeywordExpansion():
    def __init__(self, new_model=False, model_path=None, is_logging=True, tokenizer=deepcut.tokenize):
        self.new_model = new_model
        self.model_path = model_path
        self.model = None
        self.tokenizer = tokenizer

        if not self.new_model:
            if model_path is not None:
                if not os.path.exists(model_path):
                    raise ValueError("File " + model_path + " does not exist")
                self.model = gensim.models.Word2Vec.load(model_path)
            else:
                if not os.path.exists(constant.DEFAULT_MODEL_PATH):
                    raise ValueError("DEFULT_MODEL_PATH does not exist")
                self.model = gensim.models.Word2Vec.load(
                    constant.DEFAULT_MODEL_PATH)
        else:
            if model_path is not None:
                raise ValueError('model_path must be none for new model')
            self.model = None

        if is_logging:
            logging.basicConfig(
                format='%(levelname)s : %(message)s', level=logging.INFO)
            logging.root.level = logging.INFO

        self.graph = tf.get_default_graph()

    def train(self, sentence, size=constant.VECTOR_DIM, sg=constant.CBOW_SG_FLAG,
              window=constant.WINDOW, min_count=constant.MIN_COUNT, workers=constant.WORKERS):
        self.model = gensim.models.Word2Vec(
            sentence, size=size, sg=sg, window=window, min_count=min_count, workers=workers)
        self.new_model = False

    def evaluate(self, query_word, expect_word, n_expand=constant.N_EXPAND):
        pred = self.predict(query_word, n_expand)
        ans = custom_matric(expect_word, pred)
        return ans

    def predict(self, query_word, n_expand=constant.N_EXPAND):
        query_word = query_word.lower()
        with self.graph.as_default():
            query_word = self.tokenizer(query_word)
        return self.model.wv.most_similar_cosmul(positive=query_word, topn=n_expand)

    def save(self, model_path):
        if self.new_model:
            raise ValueError("model is not train")
        elif os.path.exists(model_path):
            raise ValueError("File " + model_path + " exist")
        self.model.save(model_path)
