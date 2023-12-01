"""
@author: Robert Kamunde
"""

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import scipy

from tensorflow import keras
import keras.backend as K
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import SGD
from keras.regularizers import l2

import numpy as np

import tensorflow as tf
print(tf.__version__)
print(tf.test.gpu_device_name())

K.clear_session()

n_classes = 4
img_width, img_height = 299, 299
train_data_dir = 'C:/food_classification/archive/train'
validation_data_dir = 'C:/food_classification/archive/test'
nb_train_samples = 3750 # 학습시킬 총 이미지 개수
nb_validation_samples = 1250 # 검증으로 사용할 총 이미지 개수
batch_size = 20 # 샘플 단위마다 가중치 업데이트 (현재 3750개의 이미지를 20개씩 학습시키고 3750개를 모두 채우면 1Epoch가 충족된다)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


mbv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape = (299,299,3))
x = mbv2.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)

# Dense()의 맨 첫번째 인자 학습할 음식 종류 개수로 설정
predictions = Dense(5,kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

model = Model(inputs=mbv2.input, outputs=predictions)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='best_model_3class_sept.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history.log')

history = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=10,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('model_trained.h5')
    
