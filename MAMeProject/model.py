from statistics import mode
import keras as k
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Input, add
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from data_loader import MAMeDataset
import matplotlib.pyplot as plt

# from keras.utils.vis_utils import plot_model

img_size = 256
globalAVGPooling = False
weight_decay = 1e-4
last_layer_activation = ['softmax', 'sigmoid', None][0]
num_classes = 29

def block(numConv,layer_in,n_filters,dropOutR,weight_decay):
    merge_input=layer_in
    if layer_in.shape[-1]!=n_filters:
        merge_input=Conv2D(n_filters,(1,1),padding='same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(layer_in)
    conv1=Conv2D(n_filters,(3,3),padding="same",activation="relu",kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(weight_decay))(layer_in)
    for i in range(numConv-1):
        prev=conv1
        conv1=Dropout(dropOutR)(conv1)
        conv1=Conv2D(n_filters,(3,3),padding="same",activation="relu",kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(weight_decay))(conv1)
        conv1=add([conv1,prev])
        #conv1=add([conv1,merge_input])    
    conv1=Dropout(dropOutR)(conv1)
    skip=add([conv1,merge_input])
    layer_out=MaxPooling2D(pool_size=(2,2))(skip)
    layer_out = Activation('relu')(layer_out)
    return layer_out

def create_model():
  visible = Input(shape=(256, 256, 3))
  layer=block(3,visible,16,0.2,weight_decay)
  layer=block(2,layer,32,0.3,weight_decay)
  layer=block(3,layer,64,0.3,weight_decay)
  layer=block(2,layer,128,0.3,weight_decay)
  layer=block(3,layer,256,0.3,weight_decay)
  layer=block(2,layer,256,0.3,weight_decay)
  layer=block(2,layer,512,0.3,weight_decay)

  if globalAVGPooling:
    layer=GlobalAveragePooling2D()(layer)
  else:
    layer=Flatten()(layer)

  # layer=Dense(2048)(layer)
  # layer=Dense(1024)(layer)
  layer=Dense(num_classes, activation=last_layer_activation)(layer)

  model=Model(inputs=visible,outputs=layer)

  # plot_model(model, show_shapes=True, to_file='model.png')
  return model

if __name__ == '__main__':
  create_model()