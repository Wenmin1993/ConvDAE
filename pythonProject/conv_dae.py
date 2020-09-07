import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Model
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
import numpy as np
import argparse
import wfdb


ap= argparse.ArgumentParser()
ap.add_argument('-s','--samples',type = int,default=8,help = '#number of samples to visualize when decoding')
ap.add_argument('-o','--output',type = str,default='output.png',help = 'path to output visualization file')
ap.add_argument('-p','--plot',type = str,default='plot.png',help = 'path to output plot file')
args = vars(ap.parse_args())



class ConvAutoencoder(Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([layers.Input(shape=(28,28,1)),
                                            layers.Conv1D(16,(3,3), activation='relu',padding='same',strides=2),
                                            layers.Conv1D(8,(3,3), activation='relu',padding='same',strides=2)])

        self.decoder = tf.keras.Sequential([layers.Dense(16, activation='relu'),
                                            layers.Dense(16, activation='relu'),
                                            layers.Dense(140, activation='sigmoid')])
    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = ConvAutoencoder()
autoencoder.compile(optimizer='adam',loss='mae')
history = autoencoder.fit(normal_train_data,normal_train_data,epochs=20,batchsize=512,validation_data=(test_data,test_data),shuffle=True)
