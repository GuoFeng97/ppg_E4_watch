# -*- coding: utf-8 -*-
'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import numpy as np
import keras
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
#from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def support_vector_classifier(features, labels):
    parameters = {
        'C': list(map(float, [1] + list(range(10, 101, 10)))),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }
    classifier = GridSearchCV(SVC(random_state=1, probability=True), parameters, n_jobs=-1)
    classifier.fit(features, labels)
    return classifier

def expand_single_wave_to_80(waveforms):
    expand_waveforms =[]
    length = len(waveforms)
    for i in range(0, length):
        wavelength = len(waveforms[i])
        need_expand = np.zeros(240 - wavelength)
        expand_waveforms.append(np.concatenate((waveforms[i], need_expand), axis=0))
    return np.array(expand_waveforms)

def model_evaluate_classify(model, x_test):
    # 长度统一，方便输入神经网络，统一为 64*1.2=76.8，可以统一为80
    x_test = expand_single_wave_to_80(x_test)
    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)
    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    y_new = model.predict_classes(x_test)
    return y_new

def model_evaluate_regress(model, x_test):
    # 长度统一，方便输入神经网络，统一为 64*1.2=76.8，可以统一为80
    x_test = expand_single_wave_to_80(x_test)
    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)
    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    # y_test = keras.utils.to_categorical(y_test, 10)
    model.compile(loss='mse',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy',
    #               optimizer=RMSprop(),
    #               metrics=['accuracy'])
    # score = model.evaluate(x_test, y_test, verbose=0)

    ynew = model.predict(x_test)
    return ynew

def deep_learning(x_train, y_train, x_test, y_test):
    batch_size = 128
    num_classes = 2
    # epochs = 300
    epochs = 50
    # the data, split between train and test sets
    #长度统一，方便输入神经网络，统一为 64*1.2=76.8，可以统一为80
    x_train = expand_single_wave_to_80(x_train)
    x_test = expand_single_wave_to_80(x_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # scaler = StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)
    # scaler = StandardScaler().fit(x_test)
    # x_test = scaler.transform(x_test)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
     # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(240,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    #x_train = x_train.reshape(len(x_train), -1)
    #y_train = y_train.reshape(len(y_train), -1)
    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return model,score

def lstm_method(x_train, y_train, x_test, y_test):
    num_classes = 2
    x_train = expand_single_wave_to_80(x_train)
    x_test = expand_single_wave_to_80(x_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.reshape((-1,80,1))
    x_test = x_test.reshape((-1,80,1))
    model = Sequential()
    model.add(LSTM(64, input_shape=(80, 1)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=300, batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    return model,score


def lstm_method_new(x_train, y_train, x_test, y_test):
    num_classes = 2
    x_train = expand_single_wave_to_80(x_train)
    x_test = expand_single_wave_to_80(x_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.reshape((-1,80,1))
    x_test = x_test.reshape((-1,80,1))
    model = Sequential()

    # model.add(LSTM(16))
    model.add(LSTM(16, input_shape=(80, 1)))
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu', input_shape=(80,)))
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    return model,score



def deep_learning_regress(x_train, y_train, x_test, y_test):
    batch_size = 128
    num_classes = 10
    epochs = 200


    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #长度统一，方便输入神经网络，统一为 64*1.2=76.8，可以统一为80
    x_train = expand_single_wave_to_80(x_train)
    x_test = expand_single_wave_to_80(x_test)

    #x_train = x_train.reshape(60000, 784)
    #x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    new_train_label = []
    for item in y_train:
        #设置不同的等级，0：0.05， 1：4.6，3： 2.4
        if item == '0':
            new_train_label.append(0.05)
        if item == '1':
            new_train_label.append(5.3)
        if item == '3':
            new_train_label.append(2.8)

    new_train_label = np.array(new_train_label)
    length = len(new_train_label)
    # 生成回归数据集
    scalarY = MinMaxScaler()
    scalarY.fit(new_train_label.reshape(length,1))
    y_train_new = scalarY.transform(new_train_label.reshape(length,1))

    # 定义并拟合模型
    # model = Sequential()
    # model.add(Dense(4, input_dim=2, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dense(1, activation='linear'))
    # model.compile(loss='mse', optimizer='adam')
    # model.fit(X, y, epochs=1000, verbose=0)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(80,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.summary()

    model.compile(loss='mse',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train_new,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0)

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    return model

def CNN_for_three(x_train, y_train, x_test, y_test):
    batch_size = 128
    num_classes = 2
    # epochs = 300
    epochs = 50
    # the data, split between train and test sets
    # 长度统一，方便输入神经网络，统一为 64*1.2=76.8，可以统一为80
    x_train = expand_single_wave_to_80(x_train)
    x_test = expand_single_wave_to_80(x_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.reshape((-1, 80, 1, 1))
    x_test = x_test.reshape((-1, 80, 1, 1))
    print(np.shape(x_train),np.shape(y_train))
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(80,1,1,1,), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('softmax'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return model,score