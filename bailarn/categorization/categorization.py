from __future__ import unicode_literals, print_function, division

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pickle
import time
import math
import sys
import json
from six import string_types

import warnings
warnings.filterwarnings('ignore')

import keras.models
import keras.callbacks
import numpy as np

from . import constant
from .models import CNN_XMTC_Model, save_model, load_model
from .metrics import custom_metric
from ..utils import utils

from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

from keras.optimizers import Adam
import tensorflow as tf


class Categorization(object):

    def __init__(self, tag_index=None, embedding_matrix=None, model_path=None, new_model=False):

        self.tag_index = tag_index
        if tag_index is None:
            self.tag_index = utils.build_tag_index(
                constant.TAG_LIST, constant.TAG_START_INDEX)
        self.rev_tag_index = {v: k for k, v in self.tag_index.items()}
        self.embedding_matrix = embedding_matrix

        self.model = CNN_XMTC_Model(
            embedding_matrix, len(self.tag_index)).model
        self.model.summary()
        self.new_model = new_model

        if not self.new_model:
            if model_path is not None:
                if not os.path.exists(model_path):
                    raise ValueError("File " + model_path + " does not exist")
                self.model = load_model(model_path)
            else:
                self.model = load_model(constant.DEFAULT_MODEL_PATH)
        else:  # new model
            if model_path is not None:
                raise ValueError('model_path must be none for new model')

        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def train(self, X_train, y_train, train_name="untitled", batch_size=100, learning_rate=0.001, validate_ratio=0.1,
              epochs=30, sensitive_learning=True):
        """
        Train the model on given data

        :return: History object
        """

        print("Start training Model...")

        # Optimizer
        optimizer = Adam(learning_rate)

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['top_k_categorical_accuracy'],
        )

        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=0, verbose=0, mode='auto')

#         callbacks = CustomCallback(train_name).callbacks

        if sensitive_learning:
            # compute class weight
            class_weight_y = []
            for onehot_y in y_train:
                for idx, value in enumerate(onehot_y):
                    if value == True:
                        class_weight_y.append(idx)
            class_weight_dict = dict(enumerate(class_weight.compute_class_weight(
                'balanced', np.unique(class_weight_y), class_weight_y)))

            cnn_hist = self.model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validate_ratio,
                callbacks=[early_stopping_callback],
                class_weight=class_weight_dict)
        else:
            cnn_hist = self.model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validate_ratio,
                callbacks=[early_stopping_callback])

        self.new_model = False
        return cnn_hist

    def evaluate(self, x_true, y_true, threshold_selection=constant.DEFAULT_THRESHOLD_SELECTION_PATH):
        """
        Evaluate the model on given data
        :params: threshold_selection can be number or json filepath

        :return: score object
        """

        if self.new_model:
            raise RuntimeError('The model is not being trained.')

        if y_true is None:
            raise ValueError(
                'Y can not be None, please set for_train=True in build_input.')

        print("Start evaluating...")
        with self.graph.as_default():
            y_pred = self.model.predict(x_true)

        if isinstance(threshold_selection, string_types):
            with open(threshold_selection) as json_file:
                threshold_selection = json.load(json_file)
            y_pred = np.array([[y >= threshold_selection["class_{}".format(idx)] for idx, y in enumerate(
                ys)] for ys in y_pred], dtype=np.bool_)
        else:
            y_pred = np.array([[y >= threshold_selection for y in ys]
                               for ys in y_pred], dtype=np.bool_)

        y_true = np.array(y_true, dtype=np.bool_)

        scores = custom_metric(y_true, y_pred)

        # Display score
        for metric, score in scores.items():
            print("{0}: {1:.6f}".format(metric, score))

        return scores

    def predict(self, x, threshold_selection=constant.DEFAULT_THRESHOLD_SELECTION_PATH, decode_tag=True):
        """
        Predict labels for a given vector object
        :params: threshold_selection can be number or json filepath

        :return: list of labels with corresponding confidence intervals
        """
        with self.graph.as_default():
            y_pred = self.model.predict(x)
        print("Start predicting data...")

        if decode_tag:

            if isinstance(threshold_selection, string_types):
                with open(threshold_selection) as json_file:
                    threshold_selection = json.load(json_file)
                y_pred = np.array([[y >= threshold_selection["class_{}".format(idx)] for idx, y in enumerate(
                    ys)] for ys in y_pred], dtype=np.bool_)
            else:
                y_pred = np.array([[y >= threshold_selection for y in ys]
                                   for ys in y_pred], dtype=np.bool_)

            decoded_ys_pred = []
            for ys in y_pred:
                decoded_y_pred = []
                for idx, y in enumerate(ys):
                    if y:
                        decoded_y_pred.append(self.rev_tag_index[idx])
                decoded_ys_pred.append(decoded_y_pred)
            return decoded_ys_pred
        else:
            return y_pred

    def save(self, filepath):
        """ Save the keras NN model to a HDF5 file """

        if os.path.exists(filepath):
            raise ValueError("File " + filepath + " already exists!")
        save_model(self.model, filepath)
