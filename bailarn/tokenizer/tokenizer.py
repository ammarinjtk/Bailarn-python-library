from .model import Model, save_model, load_model
from .metric import custom_metric
from . import constant
from ..utils import utils

from keras.optimizers import Adam
import tensorflow as tf
import keras

import numpy as np
import os


class Tokenizer(object):
    def __init__(self, char_index=None, tag_index=None, model_path=None, new_model=False):
        keras.backend.clear_session()
        self.char_index = char_index
        if char_index is None:
            self.char_index = utils.build_tag_index(constant.CHARACTER_LIST, constant.CHAR_START_INDEX)
        self.rev_char_index = {v: k for k, v in self.char_index.items()}
        self.tag_index = tag_index
        if tag_index is None:
            self.tag_index = utils.build_tag_index(constant.TAG_LIST, constant.TAG_START_INDEX)

        self.new_model = new_model
        self.model_path = model_path

        self.model = Model().model
        self.model.summary()

        if not self.new_model:
            if model_path is not None:
                if not os.path.exists(model_path):
                    raise ValueError("File " + model_path + " does not exist")
                self.model = load_model(model_path)
            else:
                self.model = load_model(constant.DEFAULT_MODEL_PATH)

        else: # new model
            if model_path is not None:
                raise ValueError('model_path must be none for new model')
        
        # prohibit error from django with tf backend
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def train(self, X_train, y_train, train_name="untitled", validation_split=0.1, learning_rate=0.001, epochs=100, batch_size=32, shuffle=False):
        
        # Optimizer
        optimizer = Adam(learning_rate)

        # Compile
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                      metrics=["categorical_accuracy"])
        
#         callbacks = CustomCallback(train_name).callbacks
            
        # Train model
        # handle validation_split = 0
        hist = self.model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs,
                       batch_size=batch_size, shuffle=shuffle)

        self.new_model = False
        return hist

    def predict(self, word, word_delimiter="|"):
        x = []
        readable_x = []
        for idx, char in enumerate(word):
            # compute x
            try:
                self.char_index[char]
            except KeyError:
                x.append(constant.UNKNOW_CHAR_INDEX)
            else:
                x.append(self.char_index[char])
            readable_x.append(char)
        # padding
        x = utils.pad([x], constant.SEQUENCE_LENGTH, 0)
        readable_x = utils.pad([readable_x], constant.SEQUENCE_LENGTH, constant.READABLE_PAD_CHAR)
        x = np.asarray(x)
        readable_x = np.asarray(readable_x)
        
        # Predict
        with self.graph.as_default():
            y_pred = self.model.predict(x)
        y_pred = np.argmax(y_pred, axis=2)

        # # Flatten to 1D
        y_pred = y_pred.flatten()
        readable_x = readable_x.flatten()

        # Result list
        all_result = list()
        
        # Process on each character
        for sample_idx, char in enumerate(readable_x):
            
            label = y_pred[sample_idx]

            # Pad label
            if label == constant.PAD_TAG_INDEX:
                continue
            # Pad char
            if char == constant.READABLE_PAD_CHAR:
                continue

            # Append character to result list
            all_result.append(char)

            # Skip tag for spacebar character
            if char == constant.SPACEBAR:
                continue

            # Tag at segmented point
            if label != constant.NON_SEGMENT_TAG_INDEX:
                # Append delimiter to result list
                all_result.append(word_delimiter)

        return [ word for word in "".join(all_result).split(word_delimiter) if word != ""]

    def evaluate(self, x_true, y_true):

        # Predict
        y_pred = self.model.predict(x_true)
        y_pred = np.argmax(y_pred, axis=2)
        y_true = np.argmax(y_true, axis=2)

        # Calculate score
        scores = custom_metric(y_true, y_pred)

        # Display score
        for metric, score in scores.items():
            print("{0}: {1:.6f}".format(metric, score))

        return scores

    def save(self, filepath):
        """ Save the keras model to a HDF5 file """
        if os.path.exists(filepath):
            raise ValueError("File " + filepath + " already exists!")
        save_model(self.model, filepath)
