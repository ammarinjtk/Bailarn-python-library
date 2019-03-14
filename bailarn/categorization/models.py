from keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, \
    MaxPool1D, Conv1D, Flatten, Concatenate, Embedding, Bidirectional
from keras.layers.core import Reshape
from keras.models import Model, Sequential
from keras.models import load_model as keras_load_model
from keras.engine import Layer, InputSpec
import tensorflow as tf
import math

from . import constant


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=2, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):

        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        return top_k


class DynamicKMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k_top=5, current_layer=1, total_layer=2, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k_top = k_top
        self.current_layer = current_layer
        self.total_layer = total_layer
        self.k = max(k_top, math.ceil(
            ((total_layer - current_layer) / float(total_layer)) * constant.SEQUENCE_LENGTH))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):

        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        return top_k


class CNN_XMTC_Model(object):
    """ 
    Create a keras model of a CNN XMTC.
    """
    
    def __init__(self, embedding_matrix, output_length):

        DROPOUT_RATE = 0.5
        FILTER_SIZES = [2, 4, 8]
        NUM_FILTERS = 32

        inputs = Input(shape=(constant.SEQUENCE_LENGTH,), dtype='int32')
        if embedding_matrix is None:
            embedding = Embedding(input_dim=constant.WORD_INDEXER_SIZE, output_dim=constant.EMBEDDING_SIZE, 
                                  input_length=constant.SEQUENCE_LENGTH)(inputs)
        else:
            embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=constant.EMBEDDING_SIZE,
                                  weights=[embedding_matrix], input_length=constant.SEQUENCE_LENGTH)(inputs)
        conv_0 = Conv1D(
            NUM_FILTERS, kernel_size=FILTER_SIZES[0], kernel_initializer='normal', activation='relu')(embedding)
        conv_1 = Conv1D(
            NUM_FILTERS, kernel_size=FILTER_SIZES[1], kernel_initializer='normal', activation='relu')(embedding)
        conv_2 = Conv1D(
            NUM_FILTERS, kernel_size=FILTER_SIZES[2], kernel_initializer='normal', activation='relu')(embedding)

        maxpool_0 = DynamicKMaxPooling(
            k_top=5, current_layer=1, total_layer=2)(conv_0)
        maxpool_1 = DynamicKMaxPooling(
            k_top=5, current_layer=1, total_layer=2)(conv_1)
        maxpool_2 = DynamicKMaxPooling(
            k_top=5, current_layer=1, total_layer=2)(conv_2)

        reshape_maxpool_0 = Reshape((-1, 32))(maxpool_0)
        reshape_maxpool_1 = Reshape((-1, 32))(maxpool_1)
        reshape_maxpool_2 = Reshape((-1, 32))(maxpool_2)

        conv_0_2 = Conv1D(
            NUM_FILTERS, kernel_size=FILTER_SIZES[0], kernel_initializer='normal', activation='relu')(reshape_maxpool_0)
        conv_1_2 = Conv1D(
            NUM_FILTERS, kernel_size=FILTER_SIZES[1], kernel_initializer='normal', activation='relu')(reshape_maxpool_1)
        conv_2_2 = Conv1D(
            NUM_FILTERS, kernel_size=FILTER_SIZES[2], kernel_initializer='normal', activation='relu')(reshape_maxpool_2)

        maxpool_0_2 = KMaxPooling(2)(conv_0_2)
        maxpool_1_2 = KMaxPooling(2)(conv_1_2)
        maxpool_2_2 = KMaxPooling(2)(conv_2_2)

        concatenated_tensor = Concatenate(axis=1)(
            [maxpool_0_2, maxpool_1_2, maxpool_2_2])
        dropout = Dropout(DROPOUT_RATE)(concatenated_tensor)
        flatten = Flatten()(dropout)
        hidden = Dense(64, activation='relu')(flatten)
        batch_normalization = BatchNormalization()(hidden)
        output = Dense(output_length, activation='sigmoid')(batch_normalization)

        self.model = Model(inputs=inputs, outputs=output)

def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    return keras_load_model(model_path, {'DynamicKMaxPooling': DynamicKMaxPooling, 'KMaxPooling': KMaxPooling})
