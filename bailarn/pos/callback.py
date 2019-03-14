"""
Keras Callback
"""

import csv
import os
from datetime import datetime
import numpy as np
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
import sklearn.metrics
from .metric import custom_metric
import keras.backend as K



class CustomCallback(object):
    def __init__(self, train_name, save_checkpoint_model=True):

        directory_name = datetime.today().strftime("%d-%m-%Y-%H-%M-%S")+"_"+train_name
        checkpoint_directory = os.path.join("checkpoint", directory_name)
        tensorboard_directory = os.path.join(checkpoint_directory, "tensorboard")
        
        os.makedirs(checkpoint_directory)
        os.makedirs(tensorboard_directory)
        path ={
            "checkpoint": checkpoint_directory,
            "tensorboard": tensorboard_directory,
            "loss_log": os.path.join(checkpoint_directory, "loss.csv"),
            "score_log": os.path.join(checkpoint_directory, "score.csv")
            }

        if save_checkpoint_model:
            self.callbacks = [
                ModelCheckpoint(path["checkpoint"]),
                TensorBoard(path["tensorboard"]),
                CSVLogger(path["loss_log"], ),
                CalcScore(path["score_log"])
            ]
        else:
            self.callbacks = [
                TensorBoard(path["tensorboard"]),
                CSVLogger(path["loss_log"], ),
                CalcScore(path["score_log"])
            ]
        

class TensorBoard(Callback):
    def __init__(self, log_dir='./logs',
                 write_graph=False,
                 start_steps=0,
                 batch_freq=1):
        super(TensorBoard, self).__init__()

        global tf, projector
        import tensorflow as tf
        from tensorflow.contrib.tensorboard.plugins import projector

        self.log_dir = log_dir
        self.batch_freq = batch_freq
        self.write_graph = write_graph

        self.start_steps = start_steps
        self.steps_counter = 1

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()

        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def save_scalar(self, logs):
        log = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(
                summary,
                self.start_steps + self.steps_counter
            )
        self.writer.flush()

    def on_batch_end(self, batch, logs=None):
        if self.steps_counter % self.batch_freq == 0:
            self.save_scalar(logs)
        self.steps_counter += 1

    def on_epoch_end(self, epoch, logs=None):
        self.save_scalar(logs)

    def on_train_end(self, _):
        self.writer.close()


from keras_contrib.utils import save_load_utils
class ModelCheckpoint(Callback):

    def __init__(self,
                 filepath,
                 start_steps=0,
                 batch_freq=1):
        super(ModelCheckpoint, self).__init__()

        self.filepath = filepath

        self.steps_counter = 0
        self.start_steps = start_steps
        self.batch_freq = batch_freq

        self.steps_counter

    def save_model(self):
        save_load_utils.save_all_weights(
            self.model, self.filepath+'/'+str(self.steps_counter)+'.hdf5')

    def on_batch_end(self, batch, logs=None):
        if self.steps_counter % self.batch_freq == 0:
            pass

    def on_epoch_end(self, epoch, logs=None):
        self.save_model()
        self.steps_counter += 1

class CalcScore(Callback):
    """Calculate score on custom metric with Keras callback"""

    def __init__(self, filename):
        super(CalcScore, self).__init__()
        self.file = open(filename, "w")
        self.writer = None

    def on_epoch_end(self, epoch, logs=None):
        # Validation data
        try:
            x_true = self.validation_data[0]
            y_true = self.validation_data[1]
        except IndexError:
            return 

        y_true = y_true.flatten()

        # Predict
        y_pred = self.model.predict(x_true)
        y_pred = np.argmax(y_pred, axis=2).flatten()

        scores= custom_metric(y_true, y_pred)

        # Display score
        print(end="\r")
        for metric, score in scores.items():
            print("| {0}: {1:.4f} |".format(metric, score), sep="", end="")

        # Save score to file
        if not self.writer:
            fields = ["epoch"] + sorted(scores.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=fields)
            self.writer.writeheader()

        row = scores
        row["epoch"] = epoch
        self.writer.writerow(row)
        self.file.flush()

        print("\n")

    def on_train_end(self, logs=None):
        self.file.close()
        self.writer = None
