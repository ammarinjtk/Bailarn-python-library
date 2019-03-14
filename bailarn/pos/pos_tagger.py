import json
import os
import sys


import numpy as np
import sklearn.metrics
import pandas as pd
from keras_contrib.layers import CRF

from . import constant
from .model import Model, load_model, save_model
from .metric import custom_metric
from .callback import CustomCallback

import tensorflow as tf


class POSTagger(object):

    def __init__(self, model_path=None, new_model=False, tag_index=None, embedding_matrix=None):

        self.new_model = new_model
        self.model_path = model_path
        self.model = None
        self.tag_index = tag_index
        if tag_index is None:
            print('use constant tag_indexer')
            # raise ValueError('tag_index is not define')
            self.tag_index = constant.TAG_INDEXER


        if not self.new_model:
            if model_path is not None:
                if not os.path.exists(model_path):
                    raise ValueError("File " + model_path + " does not exist")
                self.model = load_model(model_path)
            else:
                if not os.path.exists(constant.DEFULT_MODEL_PATH):
                    raise ValueError("DEFULT_MODEL_PATH does not exist")
                self.model = load_model(constant.DEFULT_MODEL_PATH)
        else:
            if model_path is not None:
                raise ValueError('model_path must be none for new model')
            self.model = Model(embedding_matrix).model
            
        # prohibit error from django with tf backend
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def evaluate(self, x_true, y_true):
        if self.new_model:
            raise ValueError("model is not trained")

        model = self.model
        
        with self.graph.as_default():
            pred = model.predict(x_true)
            
        argmax_pred = np.argmax(pred, axis=2)
        pred_flat = argmax_pred.flatten()
        true_flat = y_true.flatten()
        scores = custom_metric(true_flat, pred_flat)
        return scores


    def train(self, x_true, y_true, train_name='untitled', validation_split=0.1,epochs=100, batch_size=32, learning_rate=0.001,shuffle=False, save_checkpoint_model=True):
        """Train model"""

        callbacks = CustomCallback(train_name,save_checkpoint_model).callbacks
        
        self.model.summary()
        # Train model
        self.model.fit(x_true, y_true, validation_split=validation_split, epochs=epochs,batch_size=batch_size, shuffle=shuffle ,callbacks=callbacks)
        
        self.new_model = False

    def save(self, path):
        if self.new_model:
            raise ValueError("model is not train")
        elif os.path.exists(path):
                    raise ValueError("File " + path + " exist")
        save_model(path, self.model)

    def predict(self, x_vector, decode_tag = True):
        if self.new_model:
            raise ValueError("model is not trained")
            
        model = self.model
            
        with self.graph.as_default():
            pred = model.predict(x_vector)
        
        if decode_tag :

            original_shape = pred.shape[0:-1]
            pred_argmax = np.argmax(pred, axis=2).flatten()
            rev_dic = {}
            for k in self.tag_index:
                rev_dic[self.tag_index[k]] = k
            readable = np.array([rev_dic[e] for e in pred_argmax] )
            return readable.reshape((original_shape))

        else:
            return pred
    