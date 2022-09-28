import pickle

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from keras.layers import Dropout, Activation
from keras.optimizer_v2 import adam
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
import tensorflow as tf
import glob
import random
import cv2
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator

def applyImageAugmentationAndRetrieveGenerator(image_path, mask_path):

    data_gen_args = dict(rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2
                         )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_generator = image_datagen.flow_from_directory(image_path,
                                                        target_size=(256,256),
                                                        class_mode=None,
                                                        seed=seed,
                                                        batch_size=2)

    mask_generator = mask_datagen.flow_from_directory(mask_path,
                                                      target_size=(256, 256),
                                                      class_mode=None,
                                                      seed=seed,
                                                      batch_size=2)

    #Combine generators into one which yields image and masks
    #print(image_generator[0])
    #print(mask_generator[0])
    train_generator = zip(image_generator, mask_generator)
    return train_generator

train_generator = applyImageAugmentationAndRetrieveGenerator(r"C:\Users\amrye\PycharmProjects\pythonProject2\Train\Images", r"C:\Users\amrye\PycharmProjects\pythonProject2\Train\Masks")
test_generator = applyImageAugmentationAndRetrieveGenerator(r"C:\Users\amryePycharmProjects\pythonProject2\Validation\Images", r"C:\Users\amrye\PycharmProjects\pythonProject2\Validation\Masks")

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

def unet(sz=(256, 256, 3)):
    x = Input(sz)
    inputs = x

    # down sampling
    f = 8
    layers = []

    for i in range(0, 6):
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        layers.append(x)
        x = MaxPooling2D()(x)
        f = f * 2
    ff2 = 64

    # bottleneck
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j - 1

    # upsampling
    for i in range(0, 5):
        ff2 = ff2 // 2
        f = f // 2
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j - 1

        # classification
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    # model creation
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[mean_iou])

    return model

model = unet()

train_steps = 5 //2
test_steps = 5 //2
model.fit_generator(train_generator,
                       epochs = 10, steps_per_epoch = train_steps,validation_data = test_generator, validation_steps = test_steps, verbose = 1)


# raw = Image.open(r'C:\Users\amrye\PycharmProjects\pythonProject2\Validation\Images\Images\image5.png')
# raw = np.array(raw.resize((256, 256)))/255.
# raw = np.stack((raw,)*3, axis=-1) #convert from grayscale to rgb
# #predict the mask
# pred = model.predict(np.expand_dims(raw, 0))
#
# #mask post-processing
# msk = pred.squeeze()
# msk = np.stack((msk,)*3, axis=-1)
# msk[msk >= 0.5] = 1
# msk[msk < 0.5] = 0
#
# #show the mask and the segmented image
# combined = np.concatenate([raw, msk, raw* msk], axis = 1)
# plt.axis('off')
# plt.imshow(combined)
# plt.show()
#
# plt.imshow(plt.imread(r'C:\Users\amrye\PycharmProjects\pythonProject2\Validation\Images\Images\image6.png'), cmap="gray")
# plt.show()

# model.load_weights(r"C:\Users\amrye\PycharmProjects\pythonProject2\unet.h5")
#
# raw = Image.open('test.jpg')
# raw = np.array(raw.resize((256, 256)))/255.
# raw = raw[:,:,0:3]
#
# #predict the mask
# pred = model.predict(np.expand_dims(raw, 0))
#
# #mask post-processing
# msk  = pred.squeeze()
#
# msk = np.stack((msk,)*3, axis=-1)
# msk[msk >= 0.5] = 1
# msk[msk < 0.5] = 0
#
# #show the mask and the segmented image
# combined = np.concatenate([raw, msk, raw* msk], axis = 1)
# plt.axis('off')
# plt.imshow(combined)
# plt.show()
