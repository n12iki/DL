import tensorflow as tf
# from sklearn.utils import shuffle
import cv2
from cv2 import imread, resize
from tensorflow.keras import utils
import math
import pandas as pd
import numpy as np
import os
from collections import deque
import copy

class MAMeDataset(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, n_class, mode='train'):
        dataset = pd.read_csv('dataset/MAME_dataset.csv')
        dataste = dataset[dataset['Subset']==mode]# [['Image File', 'Medium']]
        print(dataset)
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.n_class = n_class
        
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        
        
        data_x = list()
        for batch in batch_x:
            tmp = list()
            for img_path in batch:
                try:
                    img = imread(img_path)
                    if img.shape != (192, 256, 3):
                        img = cv2.resize(img,(256, 192))
                    tmp.append(img)
                except Exception as e:
                    print(e)
                    print('failed to find path {}'.format(img_path))
            data_x.append(tmp)
        # 
        data_x = np.array(data_x, dtype='float32')
        data_y = np.array(batch_y)
        data_y = utils.to_categorical(data_y, self.n_class)
        return data_x, data_y
    
    # def on_epoch_end(self):
    #     # option method to run some logic at the end of each epoch: e.g. reshuffling
    #     seed = np.random.randint()
    #     self.x = shuffle(self.x, random_state=seed)
    #     self.y = shuffle(self.y, random_state=seed)

# class ActionDataGenerator(object):
    
#     def __init__(self,root_data_path,temporal_stride=1,temporal_length=16,resize=224, max_sample=20):
        
#         self.root_data_path = root_data_path
#         self.temporal_length = temporal_length
#         self.temporal_stride = temporal_stride
#         self.resize=resize
#         self.max_sample=max_sample

#     def file_generator(self,data_path,data_files):
#         '''
#         data_files - list of csv files to be read.
#         '''
#         for f in data_files:       
#             tmp_df = pd.read_csv(os.path.join(data_path,f))
#             label_list = list(tmp_df['Label'])
#             total_images = len(label_list) 
#             if total_images>=self.temporal_length:
#                 num_samples = int((total_images-self.temporal_length)/self.temporal_stride)+1
                
#                 img_list = list(tmp_df['FileName'])
#             else:
#                 print ('num of frames is less than temporal length; hence discarding this file-{}'.format(f))
#                 continue
            
#             samples = deque()
#             samp_count=0
#             for img in img_list:
#                 if samp_count == self.max_sample:
#                     break
#                 samples.append(img)
#                 if len(samples)==self.temporal_length:
#                     samples_c=copy.deepcopy(samples)
#                     samp_count+=1
#                     for t in range(self.temporal_stride):
#                         samples.popleft()
#                     yield samples_c,label_list[0]

#     def load_samples(self,data_cat='train', test_ratio=0.1):
#         data_path = os.path.join(self.root_data_path,data_cat)
#         csv_data_files = os.listdir(data_path)
#         file_gen = self.file_generator(data_path,csv_data_files)
#         iterator = True
#         data_list = []
#         while iterator:
#             try:
#                 x,y = next(file_gen)
#                 x=list(x)
#                 data_list.append([x,y])
#             except Exception as e:
#                 print ('the exception: ',e)
#                 iterator = False
#                 print ('end of data generator')
#         # data_list = self.shuffle_data(data_list)
#         return data_list
    
#     def train_validation_split(self, data_list, target_column, val_size=0.1, ks_sequence=False):
#         dataframe = pd.DataFrame(data_list)
#         dataframe.columns = ['Feature', target_column]
#         data_dict = dict()
#         for i in range(len(np.unique(dataframe[target_column]))):
#             data_dict[i] = dataframe[dataframe[target_column]==i]
#         train, validation = pd.DataFrame(), pd.DataFrame()
#         for df in data_dict.values():
#             cut = int(df.shape[0] * val_size)
#             val = df[:cut]
#             rem = df[cut:]
#             train = train.append(rem, ignore_index=True)
#             validation = validation.append(val, ignore_index=True)
#         if ks_sequence:
#             return train['Feature'].values.tolist(), train['Label'].values.tolist(), \
#                 validation['Feature'].values.tolist(), validation['Label'].values.tolist() # without shuffle
#         return train.values.tolist(), validation.values.tolist() # without shuffle

# root_data_path = 'C:\\Users\\AI-lab\\Documents\\activity_file\\UCF101\\csv_files\\' # machine specific
# CLASSES = 101
# BATCH_SIZE = 10
# EPOCHS = 1
# TEMPORAL_STRIDE = 8
# TEMPORAL_LENGTH = 16
# MAX_SAMPLE = 20
# HEIGHT = 192
# WIDTH = 256
# CHANNEL = 3

# data_gen_obj = ActionDataGenerator(root_data_path, temporal_stride=TEMPORAL_STRIDE, \
#                                   temporal_length=TEMPORAL_LENGTH, max_sample=MAX_SAMPLE)
# train_data = data_gen_obj.load_samples(data_cat='train')
# x_train, y_train, x_val, y_val = data_gen_obj.train_validation_split(train_data, 'Label', 0.1, True)
# r1 = MEMaDataset(x_train, y_train, BATCH_SIZE, CLASSES)
# r2 = MEMaDataset(x_val, y_val, BATCH_SIZE, CLASSES)
# print(type(r1), type(r2))

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Conv3D(10, input_shape=(TEMPORAL_LENGTH,HEIGHT,WIDTH,CHANNEL), kernel_size=(2,2,2), strides=2))
# model.add(tf.keras.layers.Conv3D(10, kernel_size=(2,3,3), strides=2))
# model.add(tf.keras.layers.Conv3D(10, kernel_size=(2,3,3), strides=2))
# model.add(tf.keras.layers.Conv3D(10, kernel_size=(2,3,3), strides=2))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(101, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

# train_history = model.fit(r1, epochs=3, steps_per_epoch=r1.__len__(), verbose=1)
# score = model.evaluate(r2, steps=5)
# print(score)