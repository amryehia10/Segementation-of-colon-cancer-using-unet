import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
import numpy as np
import keras.models
import re
import sys
import os
import base64

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from keras import backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
#sys.path.append(os.path.abspath("./model"))
from load import *
from PIL import Image
from werkzeug.utils import secure_filename

FOLDER = os.path.join('static', 'photo')
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

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FOLDER

@app.route('/')
def index_view():
    return render_template('index.html')

# def convertImage(imgData1):
# 	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
# 	with open('output.png','wb') as output:
# 	    output.write(base64.b64decode(imgstr))

model = unet()
model.load_weights(r"unet.h5")

@app.route('/predict/',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        f = request.files['filename']
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    hash_split='\\'
    f.save(f"static/photo/{f.filename.split(hash_split)[-1]}")
    raw = Image.open(f"static/photo/{f.filename.split(hash_split)[-1]}")
    raw = np.array(raw.resize((256, 256)))/255.
    raw = raw[:,:,0:3]
    #predict the mask
    pred = model.predict(np.expand_dims(raw, 0))
    #mask post-processing
    msk = pred.squeeze()
    msk = np.stack((msk,)*3, axis=-1)
    msk[msk >= 0.5] = 1
    msk[msk < 0.5] = 0

    #show the mask and the segmented image
    combined = np.concatenate([raw, msk, raw* msk], axis = 1)
    plt.imsave("static/photo/mask.jpg",combined)
    # plt.axis('off')
    # plt.imshow(combined)
    # plt.show()
    return render_template("index.html", user_image=r"/static/photo/mask.jpg")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)