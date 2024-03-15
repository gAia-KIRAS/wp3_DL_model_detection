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
from tensorflow.keras.layers import BatchNormalization, Add, Average, Concatenate, LeakyReLU, Softmax, ReLU
from tensorflow.keras.layers import Reshape, Multiply, Permute, Lambda
from tensorflow.keras.activations import sigmoid,softmax

from losses import *



## Element-wise multiply input X vs attention weight matrix

def AttentionBlock(x):
    w = x.shape[-3]
    h = x.shape[-2]
    c = x.shape[-1]
    se = GlobalAveragePooling2D()(x)            # c
    se = Dense(int(c/2), activation='selu')(se) # c/2
    se = Dense(c, activation='sigmoid')(se)     # c
    se = Reshape((1,1,c))(se)                   # 1x1xc
    se = x * se                                 #

    return se
"""
def AttentionBlock(x):
    w = x.shape[-3]
    h = x.shape[-2]
    c = x.shape[-1]
    ## channel
    tl1 = tf.math.reduce_mean(x, axis=-1)                                  # WxH
    tl1 = MultiHeadAttention(num_heads=6, key_dim=12)(tl1, tl1)            # WxH
    tl1 = sigmoid(tl1)            # WxH
    tl1 = Reshape((w,h,1))(tl1)   # WxHx1
    tl1 = Multiply()([x, tl1])                    # WxHxC * WxHx1 -> WxHxC
    ## width
    tl2 = tf.math.reduce_mean(x, axis=-2)                                   # WxC
    tl2 = MultiHeadAttention(num_heads=6, key_dim=12)(tl2, tl2)             # WxC
    tl2 = sigmoid(tl2)            # WxCx1
    tl2 = Reshape((w,1,c))(tl2)   # Wx1xC
    tl2 = Multiply()([x, tl2])                    # HxWxC * Wx1xC -> WxHxC
    ## height
    tl3 = tf.math.reduce_mean(x, axis=-3)                                    # HxC
    tl3 = MultiHeadAttention(num_heads=6, key_dim=12)(tl3, tl3)              # HxC
    tl3 = sigmoid(tl3)            # HxC
    tl3 = Reshape((1,h,c))(tl3)   # 1xHxC
    tl3 = Multiply()([x, tl3])                    # WxHxC * 1xHxC -> WxHxC
    ## average
    t = Average()([tl1,tl2,tl3]) # WxHxC

    return t
"""


## build Residual block 
def ResidualBlock(input_x, filters):
    ## breadth-wise convolutions
    x1 = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(input_x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)
    
    x2 = Conv2D(filters, kernel_size=2, strides=1, padding='same', kernel_initializer="he_normal")(input_x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)

    ## fusion branches by summing up
    x  = x1 + x2

    ## convolution + BN + activation
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    ## attention 
    x = AttentionBlock(x)
    
    return x + input_x


## build model's head
def MultiResolutionSegmentationHead(x, num_classes=2,filters=64):
    ## convolutions
    x64  = Conv2D(filters=filters, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x) # 64x64x64
    x128 = x # 128x128x640
    x256 = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x)  # 256x256x64

    ## output probabilities
    out1 = Conv2D(num_classes, 1, activation='softmax')(x256) # 256x256x2
    out2 = Conv2D(num_classes, 1, activation='softmax')(x128) # 128x128x2
    out3 = Conv2D(num_classes, 1, activation='softmax')(x64)  # 64x64x2

    return out1, out2, out3


## build Residual Attention U-Net from residual block, attention block and segmentation head
def getChalBaseline(input_shape=(128,128,14), num_classes=2, dropout_ratio=0.25):
    inputs   = Input(shape=input_shape)

    filters = 64

    # double convV
    c0 = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(inputs)
    c0 = BatchNormalization()(c0)
    c0 = ReLU()(c0)    

    c0 = Conv2D(filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(c0)
    c0 = BatchNormalization()(c0)
    c0 = ReLU()(c0)    

    # encoder
    #--- down 1
    c1 = MaxPooling2D(pool_size=(2, 2))(c0)  #64x64
    c1 = Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(c1)
    c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)    

    c1 = Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(c1)
    c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)    

    #--- down 2
    c2 = MaxPooling2D(pool_size=(2, 2))(c1)  #32x32
    c2 = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(c2)
    c2 = BatchNormalization()(c2)
    c2 = ReLU()(c2)    

    c2 = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(c2)
    c2 = BatchNormalization()(c2)
    c2 = ReLU()(c2)    

    #--- down 3
    c3 = MaxPooling2D(pool_size=(2, 2))(c2) #16x16
    c3 = Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(c3)
    c3 = BatchNormalization()(c3)
    c3 = ReLU()(c3)    

    c3 = Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(c3)
    c3 = BatchNormalization()(c3)
    c3 = ReLU()(c3)    

    #--- down 4
    c4 = MaxPooling2D(pool_size=(2, 2))(c3) #8x8xchannel ---> latent: vector??? 8x8x256 -->(flattent) 16384 -->(dense) 128 -->(dense) 16384 -->(resshape) 8x8x256
    c4 = Conv2D(1024, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(c4)
    c4 = BatchNormalization()(c4)
    c4 = ReLU()(c4)    

    c4 = Conv2D(1024, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(c4)
    c4 = BatchNormalization()(c4)
    c4 = ReLU()(c4)    
 
    ## decoder
    #---- u1
    u1 = UpSampling2D()(c4)
    u1 = Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(u1)
    u1 = BatchNormalization()(u1)
    u1 = ReLU()(u1)    

    u1 = Conv2D(512, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(u1)
    u1 = BatchNormalization()(u1)
    u1 = ReLU()(u1)    
    u1 = Concatenate(axis=-1)([u1, c3])  #16x16

    #---- u2
    u2 = UpSampling2D()(u1)
    u2 = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(u2)
    u2 = BatchNormalization()(u2)
    u2 = ReLU()(u2)    

    u2 = Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(u2)
    u2 = BatchNormalization()(u2)
    u2 = ReLU()(u2)    
    u2 = Concatenate(axis=-1)([u2, c2]) #32x32
    #---- u3
    u3 = UpSampling2D()(u2)
    u3 = Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(u3)
    u3 = BatchNormalization()(u3)
    u3 = ReLU()(u3)    

    u3 = Conv2D(128, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(u3)
    u3 = BatchNormalization()(u3)
    u3 = ReLU()(u3)    
    u3 = Concatenate(axis=-1)([u3, c1]) #64x64
    #---- u4
    u4 = UpSampling2D()(u3)
    u4 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(u4)
    u4 = BatchNormalization()(u4)
    u4 = ReLU()(u4)    

    u4 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(u4)
    u4 = BatchNormalization()(u4)
    u4 = ReLU()(u4)    
    u4 = Concatenate(axis=-1)([u4, c0]) #128x128

    ## head
    output = Conv2D(2, 1, activation='softmax')(u4) # 128x128x2


    return Model(inputs, output, name='chal_baseline')


## create full model
def getModel(pre_trained_model=None):
    ## prepare model
    network = getChalBaseline(input_shape=(128,128,14)) # adjust to achieve the desired input shape

    ## use pre-trained weights if pre-trained model is available
    if pre_trained_model is not None:
        model.set_weights(pre_trained_model.get_weights())

    ## trainable network
    network.trainable = True

    ## gradient descent
    opt = tf.keras.optimizers.Adam(learning_rate=4e-4)

    ## compile model with pre-defined losses, optimizer, metrics
    network.compile(loss=[FocalLoss(), IOULoss()], optimizer=opt, metrics=[])
    
    return network


