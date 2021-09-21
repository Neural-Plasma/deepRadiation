import numpy as np
import tensorflow as tf
import sklearn
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from os.path import join as pjoin
import matplotlib.pyplot as plt
import os.path

def dnnSim(inputs,outputs,loadModel,histplot,savedir):
    test_ratio = 0.25
    dev_ratio = 0.2

    # Prepare data
    # inputs_array = np.asarray(inputs)
    # outputs_array = np.asarray(outputs)
    inputs_array = inputs
    outputs_array = outputs

    # Split into train-dev-test sets
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(inputs_array, outputs_array, test_size=test_ratio, shuffle=False)
    Xs_train, Xs_dev, ys_train, ys_dev = train_test_split(Xs_train, ys_train, test_size=dev_ratio, shuffle=True)

    # if os.path.exists(pjoin(savedir,'deep_plasma'))==False:

    if loadModel:
        print('Model data exists. loading model ...')
        deep_approx = keras.models.load_model(pjoin(savedir,'deep_radiation'))
        # history.history contains loss information
        if histplot:
            from diagn import dnn_history_plot as hist_p
            if os.path.exists(pjoin(savedir,'history_data.npz')):
                print('Training data found. Loading data...')
                data = np.load(pjoin(savedir,'history_data.npz'))
                loss=data['loss']
                val_loss=data['val_loss']
                hist_p(loss,val_loss,savedir)
    else:
        print('Model data does not exist. Building model ...')
        # Build model
        deep_approx = keras.models.Sequential()
        # deep_approx.add(layers.Flatten())
        deep_approx.add(layers.Dense(100, input_dim=2, activation='elu'))
        deep_approx.add(layers.Dense(100, activation='selu'))
        deep_approx.add(layers.Dense(100, activation='selu'))
        deep_approx.add(layers.Dense(100, activation='relu'))
        # deep_approx.add(layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='reflect', interpolation='bilinear',seed=None, fill_value='constant'))
        deep_approx.add(layers.Dense(100, activation='relu'))
        deep_approx.add(layers.Dense(1, activation='linear'))
        print(deep_approx.output_shape)
        # Compile model
        deep_approx.compile(loss='mse', optimizer='adam')
        #FIT
        # Fit!
        history = deep_approx.fit(Xs_train, ys_train, epochs=500, batch_size=8,
                    validation_data=(Xs_dev, ys_dev),
                    callbacks=keras.callbacks.EarlyStopping(patience=100))


        deep_approx.summary()
        deep_approx.save(pjoin(savedir,'deep_radiation'))
        np.savez_compressed(pjoin(savedir,'history_data.npz'),loss=history.history['loss'][1:],val_loss=history.history['val_loss'][1:])


        # history.history contains loss information
        if histplot:
            from diagn import dnn_history_plot as hist_p
            if os.path.exists(pjoin(savedir,'history_data.npz')):
                print('Training data found. Loading data...')
                data = np.load(pjoin(savedir,'history_data.npz'))
                loss=data['loss']
                val_loss=data['val_loss']
                hist_p(loss,val_loss,savedir)
    return deep_approx
