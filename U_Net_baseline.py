import os
import tensorflow as tf

# from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, MultiHeadAttention
from tensorflow.keras.layers import BatchNormalization, Add, Average, Concatenate, LeakyReLU, Softmax
from tensorflow.keras.layers import Reshape, multiply, Permute, Lambda
from tensorflow.keras.activations import sigmoid,softmax

from losses import *


def downsampleBlock(x, filters, use_maxpool=True):
    x = Conv2D(filters, 3, padding='same', kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding='same', kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    if use_maxpool == True:
        max_pool = MaxPooling2D(pool_size=(2, 2))(x)
        return  max_pool, x
    else:
        return x


def upsampleBlock(x, y, filters):
    x = UpSampling2D()(x)
    x = Concatenate(axis=-1)([x,y]) # [1536, 768, 384, 192]
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    return x


def getUnet(input_shape=(128,128,14), num_classes=2, dropout_ratio=0.2):
    filter  = [64,128,256,512, 1024]

    input   = Input(shape=input_shape)

    # encode
    x, y128 = downsampleBlock(input, filter[0]) # 64x64x64  and 128x128x64
    x, y64  = downsampleBlock(x,     filter[1]) # 32x32x128 and 64x64x128
    x, y32  = downsampleBlock(x,     filter[2]) # 16x16x256 and 32x32x256
    x, y16  = downsampleBlock(x,     filter[3]) # 8x8x512   and 16x16x512
    x       = downsampleBlock(x,     filter[4], use_maxpool=False) # 8x8x1024

    # decode
    x = upsampleBlock(x, y16,  filter[3]) # 16x16x512
    x = upsampleBlock(x, y32,  filter[2]) # 32x32x256
    x = upsampleBlock(x, y64,  filter[1]) # 64x64x128
    x = upsampleBlock(x, y128, filter[0]) # 128x128x64
    x = Dropout(dropout_ratio)(x)

    output = Conv2D(num_classes, 1, activation='softmax')(x) # 128x128x2

    return Model(input, output, name='Unet')


## create full model
def getModel(pre_trained_model=None):
    unet = getUnet(input_shape=(128,128,14))
    if pre_trained_model is not None:
        model.set_weights(pre_trained_model.get_weights())
    unet.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate=4e-4)
    unet.compile(loss=[FocalLoss(), IOULoss()], optimizer=opt, metrics=[])
    
    return unet

