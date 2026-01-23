
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Lambda
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

def SqueezeAndExcite(inputs, ratio=8):
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = init * se
    return x
'''
def ASPP(inputs):
    """ Image Pooling """
    shape = inputs.shape
    #print(shape)
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y1 = Conv2D(256, 1, padding="same", use_bias=False)(y1)
    y1 = BatchNormalization()(y1) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

    """ 1x1 conv """
    y2 = Conv2D(256, 1, padding="same", use_bias=False)(inputs)
    y2 = BatchNormalization()(y2) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y2 = Activation("relu")(y2)

    """ 3x3 conv rate=6 """
    y3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(inputs)
    y3 = BatchNormalization()(y3) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y3 = Activation("relu")(y3)

    """ 3x3 conv rate=12 """
    y4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(inputs)
    y4 = BatchNormalization()(y4) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y4 = Activation("relu")(y4)

    """ 3x3 conv rate=18 """
    y5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(inputs)
    y5 = BatchNormalization()(y5) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(256, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y = Activation("relu")(y)

    return y
'''
def deeplab_lstm(shape):
    """ Input """
    inputs = Input(shape)

    """ Encoder """
    encoder = ResNet50(weights=None, include_top=False, input_tensor=inputs[:,:,:,:])
    image_features = encoder.get_layer("conv4_block6_out").output

    """ Atros Spatial Pyramid Pooling (ASPP) """
    aspp_inputs = image_features
    shape = aspp_inputs.shape
    x1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(aspp_inputs)
    x1 = Conv2D(256, 1, padding="same", use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(x1)

    #1x1 conv
    x2 = Conv2D(256, 1, padding="same", use_bias=False)(aspp_inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    #3x3 conv rate=6
    x3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(aspp_inputs)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)

    #3x3 conv rate=12
    x4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(aspp_inputs)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)

    #3x3 conv rate=18
    x5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(aspp_inputs)
    x5 = BatchNormalization()(x5)
    x5 = Activation("relu")(x5)

    x_a = Concatenate()([x1, x2, x3, x4, x5])
    x_a = Conv2D(256, 1, padding="same", use_bias=False)(x_a)
    x_a = BatchNormalization()(x_a)
    x_a = Activation("relu")(x_a)

    """ Decoder """
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)
    x_b = encoder.get_layer("conv2_block2_out").output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)
    
    raintide = inputs[:, :12, :8, -1]
    raintide = Reshape((12, 8))(raintide)

    lstm = LSTM(units=32*32*1, activation='relu', input_shape=(12, 8), return_sequences=True)(raintide)
    lstm = Dropout(rate=0.2)(lstm)
    lstm = LSTM(units=32*32*1, activation='relu', return_sequences=True)(lstm)
    lstm = Dropout(rate=0.2)(lstm)
    lstm = LSTM(units=32*32*1, activation='relu')(lstm)
    lstm = Dropout(rate=0.2)(lstm)
    
    lstm = Dense(activation='linear', units=16)(lstm)
    lstm = Reshape((4, 4, 1))(lstm)
    lstm = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(lstm)
    lstm = UpSampling2D((8, 8), interpolation="bilinear")(lstm)
    lstm = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(lstm)
    
    x = Concatenate()([x_a, x_b, lstm])
    x = SqueezeAndExcite(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SqueezeAndExcite(x)

    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(4, 1)(x)
    x = Activation("sigmoid")(x)


    model = Model(inputs, x)
    return model

if __name__ == "__main__":
    model = deeplab_lstm((128, 128, 11))
