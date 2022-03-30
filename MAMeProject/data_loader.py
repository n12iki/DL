import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import math
import pandas as pd
import numpy as np
import random

per_sample_normalization = True

class MAMeDataset(tf.keras.utils.Sequence):
    def __init__(self, batch_size, n_class, mode='train'):
        dataset = pd.read_csv('dataset/MAME_dataset.csv')
        dataset = dataset[dataset['Subset']==mode]# [['Image File', 'Medium']]
        self.images = list(dataset['Image file'])
        self.labels = list(dataset['Medium'])

        self.images = ['.'.join(img.split(".")[:-1]) for img in self.images]
        self.batch_size = batch_size
        self.n_class = n_class
        self.mode = mode
        
    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        data_x = []
        data_y = []
        
        for img_name in batch_x:
            try:
                file_path = f'dataset/numpy_256/{img_name}.npy'
                file = np.load(file_path, allow_pickle=True)
                
                y = np.zeros(self.n_class)
                y[file[1]] = 1

                data_x.append(file[0])
                data_y.append(y)

            except Exception as e:
                print(e)
                print('failed to find path {}'.format(file_path))
        
        data_x = np.array(data_x, dtype='float32')
        data_y = np.array(data_y)
        

        # augmentation
        if self.mode == 'train':
            seed = np.random.randint(1, 25, 1)

            data_x = tf.image.random_flip_left_right(data_x, seed)
            data_x = tf.image.random_flip_up_down(data_x, seed)
            data_x = tf.image.random_brightness(data_x, .5, seed)
            data_x = tf.image.random_contrast(data_x, 0, 2, seed)
            # data_x = tf.image.random_hue(data_x, 0.5, seed)


            data_x /= 255

            data_x = tf.image.central_crop(data_x, random.choice(np.arange(.5, 1, .1)))
            data_x = tf.image.resize(data_x, (256,256), tf.image.ResizeMethod.BILINEAR)

        else: 
            data_x /= 255

        return (data_x, data_y)
    
    def on_epoch_end(self):
        # option method to run some logic at the end of each epoch: e.g. reshuffling
        self.images, self.labels = shuffle(self.images, self.labels, random_state=42)


