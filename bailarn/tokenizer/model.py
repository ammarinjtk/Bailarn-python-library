"""
Keras Model
"""

from keras.models import Sequential
from keras.layers import Embedding, LSTM, TimeDistributed, Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.models import load_model as keras_load_model

from . import constant


class Model(object):
    def __init__(self):
        # Sequential model
        model = Sequential()

        # Embedding layer
        model.add(Embedding(constant.NUM_CHARS, constant.EMBEDDING_SIZE,
                            input_length=constant.SEQUENCE_LENGTH))

        # LSTM Layer #1
        lstm = LSTM(256, return_sequences=True, unroll=True,
                    dropout=0.1, recurrent_dropout=0.1)

        model.add(Bidirectional(lstm))
        model.add(Dropout(0.1))

        # LSTM Layer #2
        lstm = LSTM(256, return_sequences=True, unroll=True,
                    dropout=0.1, recurrent_dropout=0.1)

        model.add(Bidirectional(lstm))
        model.add(Dropout(0.1))

        # LSTM Layer #3
        lstm = LSTM(128, return_sequences=True, unroll=True,
                    dropout=0.25, recurrent_dropout=0.25)

        model.add(Bidirectional(lstm))
        model.add(Dropout(0.25))

        # RNN
        model.add(TimeDistributed(Dense(constant.NUM_TAGS, activation="softmax"),
                                  input_shape=(constant.SEQUENCE_LENGTH, 128)))


        self.model = model
        
def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    return keras_load_model(model_path)
