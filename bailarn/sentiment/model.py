"""
Keras Model
"""
import json

from keras.models import Model as keras_Model
from keras.layers import Embedding, Dense, Input, Convolution1D,MaxPooling1D, Flatten,Dropout
from keras.layers.merge import Concatenate
from . import constant

from keras import models

from keras_contrib.layers import CRF

class Model(object):
    def __init__(self, embedding_matrix=None):

        self.model_name = 'cnn-multichannel'

        model_input = Input(shape=(constant.SEQUENCE_LENGTH,))

        #set initial weights for embedding layer
        if embedding_matrix is None:
            z = Embedding(constant.WORD_INDEXER_SIZE, constant.EMBEDDING_SIZE, input_length=constant.SEQUENCE_LENGTH, name="embedding")(model_input)

        else:
            z = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=constant.SEQUENCE_LENGTH, name="embedding", weights = [embedding_matrix])(model_input)


        z = Dropout(constant.DROPOUT_PROB)(z)

        # Convolutional block
        conv_blocks = []
        for sz,nf in zip(constant.FILTER_WIDTH, constant.NUM_FILTER):
            conv = Convolution1D(filters=nf,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(constant.DROPOUT_PROB)(z)
        z = Dense(constant.HIDDEN_DIM, activation="relu")(z)
        model_output = Dense(constant.NUM_TAGS, activation="softmax")(z)

        model = keras_Model(model_input, model_output)
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])    
        self.model = model
        print(self.model_name)
        
    
def load_model(model_path):
#     from keras_contrib.utils import save_load_utils
#     model_architecture = Model()
#     model = model_architecture.model
#     save_load_utils.load_all_weights(
#         model, model_path, include_optimizer=False)
    model = models.load_model(model_path)
    return model


def save_model(model, model_path):
    # from keras_contrib.utils import save_load_utils
    # save_load_utils.save_all_weights(model, model_path)
    model.save(model_path)
