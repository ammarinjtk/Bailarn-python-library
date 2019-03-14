import numpy as np
import sys
import json
import pickle

from ..utils import utils

from ..tokenizer import constant as tokenizer_constant
from ..word_embedder import constant as word_embedder_constant
from ..pos import constant as pos_constant
from ..ner import constant as ner_constant
from ..categorization import constant as categorization_constant
from ..sentiment import constant as sentiment_constant
from ..keyword_expansion import constant as keyword_expansion_constant

from ..keyword_expansion.keyword_expansion import KeywordExpansion
from ..sentiment.analyzer import SentimentAnalyzer
from ..categorization.categorization import Categorization
from ..ner.ner import NamedEntityRecognizer
from ..word_embedder.word2vec import Word2Vec
from ..tokenizer.tokenizer import Tokenizer
from ..pos import POSTagger


tokenizer_model = Tokenizer()


def tokenize(text):
    return tokenizer_model.predict(text)


w2v_model = Word2Vec()

keyword_expansion_model = KeywordExpansion(tokenizer=tokenize)

categorization_model = Categorization()
categorization_word_index = json.load(
    open('./bailarn/categorization/catetorization_word_index.json'))
categorization_tag_index = utils.build_tag_index(
    categorization_constant.TAG_LIST, categorization_constant.TAG_START_INDEX)

sentiment_model = SentimentAnalyzer()
sentiment_word_index = pickle.load(
    open('./bailarn/sentiment/sentiment_word_index.pickle', 'rb'))
sentiment_word_index.pop('<UNK>', None)
sentiment_word_index['UNK'] = len(sentiment_word_index)
sentiment_tag_index = utils.build_tag_index(
    sentiment_constant.TAG_LIST, sentiment_constant.TAG_START_INDEX)

ner_model = NamedEntityRecognizer()
ner_word_index = pickle.load(
    open('./bailarn/ner/ner_word_index.pickle', 'rb'))
ner_word_index["<PAD>"] = 0
ner_tag_index = utils.build_tag_index(
    ner_constant.TAG_LIST, start_index=ner_constant.TAG_START_INDEX)

pos_model = POSTagger()
pos_word_index = json.load(
    open('./bailarn/pos/pos_word_index.json'))
pos_tag_index = utils.build_tag_index(
    pos_constant.TAG_LIST, start_index=pos_constant.TAG_START_INDEX)
pos_tag_index["<PAD>"] = 0


def word_embedding(text):
    return w2v_model.predict(text)


def keyword_expansion(text):
    return keyword_expansion_model.predict(text)


def categorzation(text):
    texts = utils.TextCollection(tokenize_function=tokenize)
    texts.add_text(text)
    vs = utils.build_input(texts, categorization_word_index,
                           categorization_tag_index, categorization_constant.SEQUENCE_LENGTH,
                           target='categorization', for_train=False)
    y = categorization_model.predict(vs.x, decode_tag=True,
                                     threshold_selection='./bailarn/categorization/threshold_selection.json')
    return y


def sentiment(text):
    texts = utils.TextCollection(tokenize_function=tokenize)
    texts.add_text(text)
    vs = utils.build_input(texts, sentiment_word_index,
                           sentiment_tag_index, sentiment_constant.SEQUENCE_LENGTH,
                           target='sentiment', for_train=False)
    y = sentiment_model.predict(vs.x, decode_tag=True)
    return y


def pos_tag(text):
    texts = utils.TextCollection(tokenize_function=tokenize)
    texts.add_text(text)
    word_list = texts.get_token_list(0)

    vs = utils.build_input(texts, pos_word_index,
                           pos_tag_index, pos_constant.SEQUENCE_LENGTH,
                           target='pos', for_train=False)
    # remove padding
    y_pred = pos_model.predict(vs.x, decode_tag=True).flatten()[
        0:len(word_list)]
    return y_pred


def ner(text):
    ner_texts = utils.TextCollection(tokenize_function=tokenize)
    ner_texts.add_text(text)
    word_list = ner_texts.get_token_list(0)

    vs = utils.build_input(ner_texts, ner_word_index,
                           ner_tag_index, ner_constant.SEQUENCE_LENGTH,
                           target='ner', for_train=False)
    # remove padding
    y_pred = ner_model.predict(vs.x, decode_tag=True)[0][0:len(word_list)]
    return y_pred
