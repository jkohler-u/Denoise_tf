# Libaries
import subprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow_io as tfio

package_name = 'tensorflow-io'
subprocess.check_call(["pip", "install", package_name])


'''optimized U-Net with dropout, MAE, and learning rate = 0.0006'''
input = layers.Input(shape=(128,768,1,))

# Encoder with added dropout
conv1 = layers.Conv2D(32, (3,3), activation="relu", padding="same")(input)
dropout1 = layers.Dropout(rate=0.2)(conv1)  
pool1 = layers.MaxPooling2D(3,strides =  2,padding="same")(dropout1)

conv2 = layers.Conv2D(64, 3, activation="relu", padding="same")(pool1)
dropout2 = layers.Dropout(rate=0.2)(conv2)  
pool2 = layers.MaxPooling2D(3,strides = 2, padding="same")(dropout2)

conv3 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool2)
dropout3 = layers.Dropout(rate=0.2)(conv3)  
pool3 = layers.MaxPooling2D(3,strides =  2,padding="same")(dropout3)

conv4 = layers.Conv2D(256, 3, activation="relu", padding="same")(pool3)
dropout4 = layers.Dropout(rate=0.2)(conv4)  
pool4 = layers.MaxPooling2D(3,strides = 2, padding="same")(dropout4)

conv5 = layers.Conv2D(512, 3, activation="relu", padding="same")(pool4)
dropout5 = layers.Dropout(rate=0.2)(conv5)  
pool5 = layers.MaxPooling2D(3,strides =  2,padding="same")(dropout5)

conv6 = layers.Conv2D(1024, 3, activation="relu", padding="same")(pool5)
dropout6 = layers.Dropout(rate=0.2)(conv6)  
pool6 = layers.MaxPooling2D(3,strides = 2, padding="same")(dropout6)

conv7 = layers.Conv2D(1024, 3, activation="relu", padding="same")(pool6)
dropout7 = layers.Dropout(rate=0.2)(conv7)  
pool7 = layers.MaxPooling2D(3,strides =  1,padding="same")(dropout7)

conv8 = layers.Conv2D(1024, 3, activation="relu", padding="same")(pool7)
dropout8 = layers.Dropout(rate=0.2)(conv8)  
pool8 = layers.MaxPooling2D(3,strides = 1, padding="same")(dropout8)

# Bridge
conv9 = layers.Conv2D(126, 3, activation="relu", padding="same")(pool8)

# Decoder
upconv1 = layers.Conv2DTranspose(1024, 3, strides=1, activation="relu", padding="same")(conv9)
concat1 = layers.Concatenate()([upconv1, dropout8])
convD1 = layers.Conv2D(1024, 3, activation="relu", padding="same")(concat1)

upconv2 = layers.Conv2DTranspose(1024, 3, strides=1, activation="relu", padding="same")(convD1)
concat2 = layers.Concatenate()([upconv2, dropout7])
convD2 = layers.Conv2D(1024, 3, activation="relu", padding="same")(concat2)
                       
upconv3 = layers.Conv2DTranspose(1024, 3, strides=2, activation="relu", padding="same")(convD2)
concat3 = layers.Concatenate()([upconv3, dropout6])
convD3 = layers.Conv2D(1024, 3, activation="relu", padding="same")(concat3)

upconv4 = layers.Conv2DTranspose(512, 3, strides=2, activation="relu", padding="same")(convD3)
concat4 = layers.Concatenate()([upconv4, dropout5])
convD4 = layers.Conv2D(512, 3, activation="relu", padding="same")(concat4)

upconv5 = layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(convD4)
concat5 = layers.Concatenate()([upconv5, dropout4])
convD5 = layers.Conv2D(256, 3, activation="relu", padding="same")(concat5)

upconv6 = layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(convD5)
concat6 = layers.Concatenate()([upconv6, dropout3])
convD6 = layers.Conv2D(128, 3, activation="relu", padding="same")(concat6)

upconv7 = layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(convD6)
concat7 = layers.Concatenate()([upconv7, dropout2])
convD7 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat7)

upconv8 = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(convD7)
concat8 = layers.Concatenate()([upconv8, dropout1])
convD8 = layers.Conv2D(32, 3, activation="relu", padding="same")(concat8)

# Output
output = layers.Conv2D(1, 3, activation="tanh", padding="same")(convD8)

# U-Net model
optimized = Model(inputs=input, outputs=output)
adam = keras.optimizers.Adam(learning_rate=0.0006)
optimized.compile(optimizer=adam, loss="mean_absolute_error")


def signalToNoiseLoss(y_true, y_pred):
    '''calculates the loss between signal and noise
    
    Args: 
      y_true: target values for signal as tensors or other array-like objects
      y_pred: predicted values for signal as tensor or other array-like objects
    Returns:
      The signal-to-noise-loss as float
    '''
    noise = y_true - y_pred
    signal = y_true
    snr = tf.math.reduce_mean(tf.math.square(signal)) / tf.math.reduce_mean(tf.math.square(noise))
    loss = -10.0 * tf.math.log(snr) / tf.math.log(10.0)
    return loss

def noiseToSignalRatio(y_true, y_pred):
    '''calculates the loss between noise and signal
    
    Args: 
      y_true: target values for signal as tensors or other array-like objects
      y_pred: predicted values for signal as tensor or other array-like objects
    Returns:
      The noise-to-signal-loss as float
    '''
    noise = y_true - y_pred
    signal = y_true
    nsr = tf.math.reduce_mean(tf.math.square(noise)) / tf.math.reduce_mean(tf.math.square(signal))
    return nsr
