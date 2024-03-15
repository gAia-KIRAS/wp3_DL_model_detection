import os
import tensorflow as tf

# from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, SeparableConv2D, UpSampling2D, UpSampling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, MultiHeadAttention
from tensorflow.keras.layers import BatchNormalization, Add, Average, Concatenate, LeakyReLU, Softmax
from tensorflow.keras.layers import Reshape, Multiply, Permute, Lambda
from tensorflow.keras.activations import sigmoid,softmax

from losses import *


def AttentionBlock(x):
    w = x.shape[-3]
    h = x.shape[-2]
    c = x.shape[-1]
    se = GlobalAveragePooling2D()(x)            # c
    se = Dense(int(c/2), activation='selu')(se) #c/2
    se = Dense(c, activation='sigmoid')(se)     # c
    se = Reshape((1,1,c))(se)                   # 1x1xc
    se = x * se                                 #

    return se



def ResidualBlock(input_x, filters):
    x1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(input_x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    x2 = Conv2D(filters, kernel_size=2, strides=1, padding='same', kernel_initializer="he_normal")(input_x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)

    x  = x1 + x2

    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = AttentionBlock(x)

    return x + input_x


def getRNet(input_shape=(128,128,23), num_classes=2, dropout_ratio=0.25):
    inputs   = Input(shape=input_shape)

    filters = 40

    c0 = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(inputs)
    c0 = BatchNormalization()(c0)
    c0 = LeakyReLU()(c0)

    # encoder
    c1 = ResidualBlock(c0, filters) # 128
    p1 = Conv2D(filters, strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(c1)

    c2 = ResidualBlock(p1, filters)     # 128
    p2 = Conv2D(filters, strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(c2)

    c3 = ResidualBlock(p2, filters)     # 128
    p3 = Conv2D(filters, strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(c3)

    c4 = ResidualBlock(p3, filters)     # 128
    p4 = Conv2D(filters, strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(c4)

    # bottom
    b = ResidualBlock(p4, filters)      # 128

    # decoder
    u4 = Conv2DTranspose(filters=b.shape[-1], strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(p4)   #  128
    u4 = Concatenate(axis=-1)([u4, c4])  # 256
    c5 = ResidualBlock(u4, 2*filters)

    u3 = Conv2DTranspose(filters=c5.shape[-1], strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(c5)    # 256
    u3 = Concatenate(axis=-1)([u3, c3])  # 381
    c6 = ResidualBlock(u3, 3*filters)

    u2 = Conv2DTranspose(filters=c6.shape[-1], strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(c6)    # 384 
    u2 = Concatenate(axis=-1)([u2, c2]) #  512 
    c7 = ResidualBlock(u2, 4*filters)

    u1 = Conv2DTranspose(filters=c7.shape[-1], strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(c7) #  512
    u1 = Concatenate(axis=-1)([u1, c1])
    x  = ResidualBlock(u1, 5*filters)

    ## multi-resolution segmentation head
    x = Dropout(dropout_ratio)(x)

    x64  = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x) # 64x64x64
    x128 = x # 128x128x640
    x256 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x)  # 256x256x64

    out1 = Conv2D(num_classes, 1, activation='softmax')(x256) # 256x256x2
    out2 = Conv2D(num_classes, 1, activation='softmax')(x128) # 128x128x2
    out3 = Conv2D(num_classes, 1, activation='softmax')(x64)  # 64x64x2

    return Model(inputs, [out1,out2,out3], name='RNet')


## create full model
def getModel(pre_trained_model=None):
    network = getRNet(input_shape=(128,128,23))
    if pre_trained_model is not None:
        model.set_weights(pre_trained_model.get_weights())
    network.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate=4e-4)
    network.compile(loss=[FocalLoss(), IOULoss()], optimizer=opt, metrics=[])

    return network


