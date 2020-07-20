from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
import load_data
from collections import OrderedDict
from sklearn import metrics
import numpy as np

EPOCHS = 100
TEST_LOCAL = False


def auc(y_true, y_pred):
    auc = tf.compat.v1.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def build_model():
  model = keras.Sequential([
    layers.Dense(30, activation='relu', input_shape=[len(train_data.keys())]),
    layers.Dense(15, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0005)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=[auc])
  return model

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


train_data, test_data = load_data.get_data()
train_data = train_data.sample(frac=1)
if TEST_LOCAL:
  dataset = train_data.copy()
  train_data = dataset.sample(frac=1)
  test_data = dataset.drop(train_data.index)
  test_labels = test_data.pop('Delay.Indicator')
train_labels = train_data.pop('Delay.Indicator')
train_stats = train_data.describe()
train_stats = train_stats.transpose()
if not TEST_LOCAL:
  submission_id  = test_data.pop('ID')

model = build_model()

normed_train_data = norm(train_data)
normed_test_data = norm(test_data)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

validation = 0.2 if TEST_LOCAL else 0

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split=validation, verbose=1)

test_predictions = model.predict(normed_test_data).flatten()
test_predictions = test_predictions.round(0)

if TEST_LOCAL:
  fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_predictions, pos_label=1)
  accuracy = metrics.auc(fpr, tpr)
  print(accuracy)

if not TEST_LOCAL:
  submission_data = pd.DataFrame(
    OrderedDict(
      {
        'ID' : pd.Series(submission_id),
        'Delay.Indicator' : pd.Series(test_predictions)
      }
    )
  )
  convert_dict = {
    'ID' : int,
    'Delay.Indicator' : int
  }
  submission_data.astype(convert_dict)
  submission_data.round(0)
  submission_data.to_csv('submission.csv', index=False)