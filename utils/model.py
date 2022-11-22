import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import cv2
from glob import glob
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import sys
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, AveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from __future__ import division
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
#from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from tensorflow.keras.optimizers import SGD,Adam
from keras.optimizers import *
from keras.layers import *        
from keras.applications.vgg16 import VGG16
import keras
import glob
from tensorflow.keras.layers.experimental import preprocessing

def output_block(inputs):
    x = Conv2D(1, (1, 1), padding="same")(inputs)
    x = Activation('sigmoid')(x)
    return x

def ASPP(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def conv_block(inputs, filters):
    x = inputs

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x
class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self, filters: int,
        kernel_size: tuple,
        strides: tuple = (1, 1),
        padding: str = "same",
        dilation_rate: tuple = (1, 1),
    ):
        super(ConvModule, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate

        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate
        )

        self.bn = BatchNormalization()

        self.relu = tf.keras.layers.ReLU()
        self.glb = GlobalAveragePooling2D()
        self.den1 = Dense(self.filters // 8, activation='relu', kernel_initializer='he_normal', use_bias=False)
        self.den2 = Dense(self.filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        init = x
        channel_axis = -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)
        se = self.glb(init)
        se = Reshape(se_shape)(se)
        se = self.den1(se)
        se = self.den2(se)
        x = Multiply()([init, se])

        return x

class PASPP(tf.keras.layers.Layer):
    def __init__(self, filters: int
    ):
        super(PASPP, self).__init__()
        self.filters = filters
        self.conv1 = Conv2D(self.filters//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")
        self.conv2 = Conv2D(self.filters//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")
        self.conv3 = Conv2D(self.filters//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")
        self.conv4 = Conv2D(self.filters//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")
        self.nhanh1 = Conv2D(self.filters//4, (3,3), dilation_rate=1, kernel_initializer='he_normal',padding="same", use_bias=False)
        self.nhanh2 =Conv2D(self.filters//4, (3,3), dilation_rate=2, kernel_initializer='he_normal',padding="same", use_bias=False)
        self.nhanh3 =Conv2D(self.filters//4, (3,3), dilation_rate=4, kernel_initializer='he_normal',padding="same", use_bias=False)
        self.nhanh4=Conv2D(self.filters//4, (3,3), dilation_rate=8, kernel_initializer='he_normal',padding="same", use_bias=False)
        self.tichchap12 = Conv2D(self.filters//2, (1,1), strides=1, kernel_initializer='he_normal', padding="same")
        self.tichchap34 = Conv2D(self.filters//2, (1,1), strides=1, kernel_initializer='he_normal', padding="same")
        self.final = Conv2D(self.filters, (1,1), strides=1, kernel_initializer='he_normal', padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        #self.bn1 = BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
    def call(self, x: tf.Tensor) -> tf.Tensor:       
     x1 = self.conv1(x)
     x1 = self.bn(x1)
     x1 = self.relu(x1)
  
  #nhanh 2
     x2 = self.conv2(x)
     x2 = self.bn(x2)
     x2 = self.relu(x2)

  #nhanh 3
     x3 = self.conv3(x)
     x3 = self.bn(x3)
     x3 = self.relu(x3)

  #nhanh 4
     x4 = self.conv4(x)
     x4 = self.bn(x4)
     x4 = self.relu(x4)

  #add 1 vs 2, 3 vs 4
     x1_2 = Add()([x1,x2])
     x3_4 = Add()([x3,x4])

  #nhanh 1
     x1 = self.nhanh1(x1)
     x1 = self.bn(x1)
     x1 = self.relu(x1)
     x1 = Add()([x1,x1_2])

  #nhanh 2
     x2 = self.nhanh2(x2)
     x2 = self.bn(x2)
     x2 = self.relu(x2)
     x2 = Add()([x2,x1_2])

  #nhanh 3
     x3 = self.nhanh3(x3)
     x3 = self.bn(x3)
     x3 = self.relu(x3)
     x3 = Add()([x3,x3_4])

  #nhanh 4
     x4 = self.nhanh4(x4)
     x4 = self.bn(x4)
     x4 = self.relu(x4)
     x4 = Add()([x4,x3_4])

  #Concatenate 1 vs 2, 3 vs4
     x1_2 = Concatenate()([x1,x2])
     x3_4 = Concatenate()([x3,x4])

  #Tich chap C//2
     x1_2 = self.tichchap12(x1_2)
     #x1_2 = self.bn(x1_2)
     x1_2 = self.relu(x1_2)

     x3_4 = self.tichchap34(x3_4)
     #x3_4 = self.bn(x3_4)
     x3_4 = self.relu(x3_4)

  #Concatenate
     y = Concatenate()([x1_2,x3_4])
     y = self.final(y)
     #y = self.bn(y)
     y = self.relu(y)

     return y  

class RFB(tf.keras.layers.Layer):
    def __init__(self, filters: int, name: str):
        super(RFB, self).__init__(name=name)
        self.filters = filters

        self.branch_1 = tf.keras.Sequential([
            ConvModule(filters=self.filters, kernel_size=(1, 1))
        ])
        self.branch_2 = tf.keras.Sequential([
            ConvModule(filters=self.filters, kernel_size=(1, 1)),
            ConvModule(filters=self.filters, kernel_size=(1, 3)),
            ConvModule(filters=self.filters, kernel_size=(3, 1)),
            ConvModule(filters=self.filters, kernel_size=(
                3, 3), dilation_rate=(3, 3))
        ])

        self.branch_3 = tf.keras.Sequential([
            ConvModule(filters=self.filters, kernel_size=(1, 1)),
            ConvModule(filters=self.filters, kernel_size=(1, 5)),
            ConvModule(filters=self.filters, kernel_size=(5, 1)),
            ConvModule(filters=self.filters, kernel_size=(
                3, 3), dilation_rate=(5, 5))
        ])

        self.branch_4 = tf.keras.Sequential([
            ConvModule(filters=self.filters, kernel_size=(1, 1)),
            ConvModule(filters=self.filters, kernel_size=(1, 7)),
            ConvModule(filters=self.filters, kernel_size=(7, 1)),
            ConvModule(filters=self.filters, kernel_size=(
                3, 3), dilation_rate=(7, 7))
        ])

        self.concate_branch = ConvModule(
            filters=self.filters, kernel_size=(1, 1))

        self.shortcut_branch = ConvModule(
            filters=self.filters, kernel_size=(1, 1))

        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.nn.relu(inputs)
        x1 = self.branch_1(inputs)
        x2 = self.branch_2(inputs)
        x3 = self.branch_3(inputs)
        x4 = self.branch_4(inputs)
        x_res = self.shortcut_branch(inputs)

        x_con = tf.concat([x1, x2, x3, x4], axis=-1)

        x_concat_conv = self.concate_branch(x_con)

        x = self.relu(x_concat_conv + x_res)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

class PartialDecoder(tf.keras.layers.Layer):
    def __init__(self, filters: int, name:str):
        super(PartialDecoder, self).__init__(name=name)
        self.filters = filters

        self.upsampling = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear')
        self.conv_up1 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up2 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up3 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up4 = ConvModule(filters=filters, kernel_size=(3, 1))
        self.conv_up41 = ConvModule(filters=filters, kernel_size=(1,3))
        self.conv_up5 = ConvModule(filters=filters, kernel_size=(3, 1))
        self.conv_up51 = ConvModule(filters=filters, kernel_size=(1,3))
        self.conv_up6 = ConvModule(filters=filters, kernel_size=(3, 1))
        self.conv_up61 = ConvModule(filters=filters, kernel_size=(1,3))

        self.conv_up7 = ConvModule(filters=filters, kernel_size=(3, 1))
        self.conv_up71 = ConvModule(filters=filters, kernel_size=(1,3))
        self.conv_up8 = ConvModule(filters=filters, kernel_size=(3, 1))
        self.conv_up81 = ConvModule(filters=filters, kernel_size=(1,3))
        self.conv_up9 = ConvModule(filters=filters, kernel_size=(3, 1))
        self.conv_up91 = ConvModule(filters=filters, kernel_size=(1,3))

    def call(self, rfb_feat1: tf.Tensor, rfb_feat2: tf.Tensor, rfb_feat3: tf.Tensor) -> tf.Tensor:
        rfb_feat1 = tf.nn.relu(rfb_feat1)
        rfb_feat2 = tf.nn.relu(rfb_feat2)
        rfb_feat3 = tf.nn.relu(rfb_feat3)

        x1_1 = self.upsampling(self.upsampling(rfb_feat1))
        x2_1 = self.upsampling(rfb_feat2) 
        xnew = tf.concat([x1_1,x2_1,rfb_feat3],-1)

        xnew1 = self.conv_up7(xnew)
        xnew1 = self.conv_up71(xnew1)
        xnew11 = self.conv_up8(xnew1)
        xnew11 = self.conv_up81(xnew11)
        xnew12 = self.conv_up9(xnew11)
        xnew12 = self.conv_up91(xnew12)
        xfinal1 = tf.concat([xnew1,xnew11,xnew12],-1)

        xnew2 = self.conv_up4(xnew)
        xnew2 = self.conv_up41(xnew2)
        xnew21 = self.conv_up5(xnew2)
        xnew21 = self.conv_up51(xnew21)
        xnew22 = self.conv_up6(xnew21)
        xnew22 = self.conv_up61(xnew22)
        xfinal2 = tf.concat([xnew2,xnew21,xnew22],-1)

        return xfinal1 + xfinal2
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

class FinalDecoder(tf.keras.layers.Layer):
    def __init__(self, filters: int, name:str):
        super(FinalDecoder, self).__init__(name=name)
        self.filters = filters

        self.upsampling = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear')
        self.conv_up1 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up2 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up3 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up4 = ConvModule(filters=filters, kernel_size=(3, 3))
        self.conv_up5 = ConvModule(filters=2*filters, kernel_size=(3, 3))

        self.conv_concat_1 = ConvModule(filters=2*filters, kernel_size=(3, 3))
        self.conv_concat_2 = ConvModule(filters=3*filters, kernel_size=(3, 3))

        self.conv4 = ConvModule(filters=2*filters, kernel_size=(3, 3))
        self.conv5 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))

    def call(self, rfb_feat1: tf.Tensor, rfb_feat2: tf.Tensor) -> tf.Tensor:
        rfb_feat1 = tf.nn.relu(rfb_feat1)
        rfb_feat2 = tf.nn.relu(rfb_feat2)


        x1_1 = rfb_feat1
        x2_1 = self.conv_up1(self.upsampling(rfb_feat1)) * rfb_feat2

        x2_2 = tf.concat([x2_1, self.conv_up4(
            self.upsampling(x1_1))], axis=-1)
        x2_2 = self.conv_concat_1(x2_2)

        x = self.conv4(x2_2)
        x = self.conv5(x)

        return x
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def paspp(x, filter):
  
  x1 = Conv2D(filter//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x)
  x1 = BatchNormalization()(x1)
  x1 = Activation("relu")(x1)
  x1 = squeeze_excite_block(x1)
  #nhanh 2
  x2 = Conv2D(filter//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x)
  x2 = BatchNormalization()(x2)
  x2 = Activation("relu")(x2)
  x2 = squeeze_excite_block(x2)
  #nhanh 3
  x3 = Conv2D(filter//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x)
  x3 = BatchNormalization()(x2)
  x3 = Activation("relu")(x2)
  x3 = squeeze_excite_block(x3)
  #nhanh 4
  x4 = Conv2D(filter//4, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x)
  x4 = BatchNormalization()(x4)
  x4 = Activation("relu")(x4)
  x4 = squeeze_excite_block(x4)
  #add 1 vs 2, 3 vs 4
  x1_2 = Add()([x1,x2])
  x3_4 = Add()([x3,x4])

  #nhanh 1
  x1 = Conv2D(filter//4, (3,3), dilation_rate=1, kernel_initializer='he_normal',padding="same", use_bias=False)(x1)
  x1 = BatchNormalization()(x1)
  x1 = Activation("relu")(x1)
  x1 = squeeze_excite_block(x1)
  x1 = Add()([x1,x1_2])
  
  #nhanh 2
  x2 = Conv2D(filter//4, (3,3), dilation_rate=2, kernel_initializer='he_normal',padding="same", use_bias=False)(x2)
  x2 = BatchNormalization()(x2)
  x2 = Activation("relu")(x2)
  x2 = squeeze_excite_block(x2)
  x2 = Add()([x2,x1_2])

  #nhanh 3
  x3 = Conv2D(filter//4, (3,3), dilation_rate=4, kernel_initializer='he_normal',padding="same", use_bias=False)(x3)
  x3 = BatchNormalization()(x3)
  x3 = Activation("relu")(x3)
  x3 = squeeze_excite_block(x3)
  x3 = Add()([x3,x3_4])

  #nhanh 4
  x4 = Conv2D(filter//4, (3,3), dilation_rate=8, kernel_initializer='he_normal',padding="same", use_bias=False)(x4)
  x4 = BatchNormalization()(x4)
  x4 = Activation("relu")(x4)
  x4 = squeeze_excite_block(x4)
  x4 = Add()([x4,x3_4])

  #Concatenate 1 vs 2, 3 vs4
  x1_2 = Concatenate()([x1,x2])
  x3_4 = Concatenate()([x3,x4])

  #Tich chap C//2
  x1_2 = Conv2D(filter//2, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x1_2)
  x1_2 = BatchNormalization()(x1_2)
  x1_2 = Activation("relu")(x1_2)
  x1_2 = squeeze_excite_block(x1_2)

  x3_4 = Conv2D(filter//2, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(x3_4)
  x3_4 = BatchNormalization()(x3_4)
  x3_4 = Activation("relu")(x3_4)
  x3_4 = squeeze_excite_block(x3_4)
  #Concatenate
  y = Concatenate()([x1_2,x3_4])
  y = Conv2D(filter, (1,1), strides=1, kernel_initializer='he_normal', padding="same")(y)
  y = BatchNormalization()(y)
  y = Activation("relu")(y)
  y = squeeze_excite_block(y)

  return y

def conv_block1(x, n_filter, pool=True):
            x = Conv2D(n_filter, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(n_filter, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = squeeze_excite_block(x)
            c = x

            if pool == True:
                x = MaxPooling2D((2, 2), (2, 2))(x)
                return c, x
            else:
                return c

class FE_backbone():
    def __init__(self, model_architecture: str = 'resnet50', inshape: tuple = (352, 352, 3), is_trainable: bool = True):
        self.inshape = inshape
        self._supported_arch = ['resnet50', 'mobilenetv2']
        self.model_architecture = model_architecture
        if self.model_architecture not in self._supported_arch:
            tf.print(
                f"Model Architecture should be one of {self._supported_arch}")
            sys.exit()

        self.is_trainable = is_trainable
        self.resnet_feature_extractor_layer_name = [
            #'block1b_add',                                       
            'block2d_add',  # 256
            'block4a_expand_activation',  # 512
            'block6a_expand_activation',  # 1024
            'top_activation',  # 2048
        ]
        self.mobilenet_feature_extractor_layer_name = [
            'block_3_expand_relu',  # 56x56x144
            'block_6_expand_relu',  # 28x28x192
            'block_13_expand_relu',  # 14x14x576
            'out_relu',             # 7x7x1280
        ]

        if self.model_architecture == 'resnet50':
            self.backbone = tf.keras.applications.EfficientNetV2S(
                include_top=False, input_shape=self.inshape
            )

        if self.model_architecture == 'mobilenetv2':
            self.backbone = MobileNetV2(
                include_top=False, input_shape=self.inshape
            )

        self.backbone.trainable = self.is_trainable

    def get_fe_backbone(self) -> tf.keras.Model:

        layer_out = []
        if self.model_architecture == 'resnet50':
            for layer_name in self.resnet_feature_extractor_layer_name:
                layer_out.append(self.backbone.get_layer(layer_name).output)

        if self.model_architecture == 'mobilenetv2':
            for layer_name in self.mobilenet_feature_extractor_layer_name:
                layer_out.append(self.backbone.get_layer(layer_name).output)

        fe_backbone_model = tf.keras.models.Model(
            inputs=self.backbone.input, outputs=layer_out, name='resnet50_include_top_false' if self.model_architecture == 'resnet50' else 'mobilenetv2_include_top_false')

        return fe_backbone_model
def build_model(shape):
  inputs = Input(shape)
  features = FE_backbone(model_architecture='resnet50',inshape=(256,256, 3), is_trainable=True ).get_fe_backbone()(inputs)

  p5 = features[3]
  p4 = features[2]
  p3 = features[1]

  a5 = paspp(p5,32)
  a4 = paspp(p4,32)
  a3 = paspp(p3,32)

  a2 = RFB(32,'2')(p5)
  a1 = RFB(32,'1')(p4)
  a0 = RFB(32,'0')(p3)
  a5 = Concatenate()([a5,a2])
  a4 = Concatenate()([a4,a1])
  a3 = Concatenate()([a3,a0])
  out1 = PartialDecoder(64,'6')(a5,a4,a3)     

  output1 = preprocessing.Resizing(256,256, name='salient_out_5')(out1)

  outputs = Conv2D(1,(1,1),activation='sigmoid')(output1)

  return Model(inputs,outputs)

if __name__ == '__main__':

   model = build_model((256,256, 3))
   print(model(np.random.rand(1,256,256,3)).shape)