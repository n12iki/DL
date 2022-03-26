from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense
from keras.layers import Flatten, Input, add, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D,Flatten,Dense
from keras import backend as K
from keras import regularizers
#from keras.utils.vis_utils import plot_model


def block(numConv,layer_in,n_filters,dropOutR,weight_decay):
    merge_input=layer_in
    if layer_in.shape[-1]!=n_filters:
        merge_input=Conv2D(n_filters,(1,1),padding='same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay))(layer_in)
    conv1=Conv2D(n_filters,(3,3),padding="same",activation="relu",kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(weight_decay))(layer_in)
    for i in range(numConv-1):
        conv1=Conv2D(n_filters,(3,3),padding="same",activation="relu",kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(weight_decay))(conv1)
    drop=Dropout(dropOutR)(conv1)
    skip=add([drop,merge_input])
    layer_out=MaxPooling2D(pool_size=(2,2))(skip)
    layer_out = Activation('relu')(layer_out)
    return layer_out

def createModel(weight_decay,num_classes,globalAVGPooling):
    visible = Input(shape=(256, 256, 3))
    layer=block(1,visible,64,0.2,weight_decay)
    layer=block(2,layer,128,0.3,weight_decay)
    layer=block(3,layer,256,0.4,weight_decay)

    if globalAVGPooling:
      layer=GlobalAveragePooling2D()(layer)
    else:
      layer=Flatten()(layer)
    layer=Dense(num_classes, activation='softmax')(layer)

    model=Model(inputs=visible,outputs=layer)
    model.summary()
    #plot_model(model, show_shapes=True, to_file='residual_module.png')
    return model

def main():
    weight_decay = 1e-4
    num_classes = 20
    globalAVGPooling = False
    model=createModel(weight_decay,num_classes,globalAVGPooling)

if __name__ == "__main__":
    main()
