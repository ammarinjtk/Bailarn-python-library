"""
Keras Model
"""

from keras.models import Sequential
from keras.layers import Embedding, LSTM, TimeDistributed, Dense, Dropout, Bidirectional
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras_contrib import metrics, losses
from . import constant

from keras_contrib.layers import CRF

class Model(object):
    def __init__(self, embedding_matrix=None):

        self.model_name = 'bi-lstm-crf'

        model = Sequential()
        
        if embedding_matrix is None:
            model.add(Embedding(constant.NUM_WORD, constant.VECTOR_DIM))  # Random embedding
        else:
            model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix]))
        model.add(Bidirectional(LSTM(218, return_sequences=True)))
        crf = CRF(constant.NUM_TAGS, sparse_target=True)
        model.add(crf)
        model.summary()
        model.compile('adam', loss=losses.crf_loss, metrics=[metrics.crf_accuracy])
        self.model = model


def load_model(model_path):
    from keras_contrib.utils import save_load_utils
    model = Model().model
    save_load_utils.load_all_weights(model, model_path, include_optimizer=False)
    
    return model


def save_model(model_path, model):
    model.save(model_path)
