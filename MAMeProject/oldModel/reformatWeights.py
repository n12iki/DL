from tensorflow.keras import optimizers
import tensorflow as tf
from data_loaderR import MAMeDataset
import matplotlib.pyplot as plt
from modelR import createModel
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json


img_size = 256
globalAVGPooling = False
num_classes = 20
weight_decay = 1e-4
loss = ['categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error', 'mean_absolute_error'][0]
n_epochs = 150
batch_size = 32
num_classes = 29

model = createModel(weight_decay,num_classes,globalAVGPooling)
latest = tf.train.latest_checkpoint("modelR1.data-00000-of-00001")
model.load_weights(latest)
model.save_weights("bestWeightsR1.hdf5")