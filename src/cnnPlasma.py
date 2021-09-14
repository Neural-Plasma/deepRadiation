import numpy as np
import tensorflow as tf
import sklearn
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from os.path import join as pjoin
import matplotlib.pyplot as plt

def dnnSim(inputs,outputs,loadModel,histplot,savedir):
    test_ratio = 0.25
    dev_ratio = 0.2

    # Prepare data
    # inputs_array = np.asarray(inputs)
    # outputs_array = np.asarray(outputs)
    inputs_array = inputs
    outputs_array = outputs

    # Split into train-dev-test sets
    X_train, X_test, y_train, y_test = train_test_split(inputs_array, outputs_array, test_size=test_ratio, shuffle=False)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=dev_ratio, shuffle=False)

    # if os.path.exists(pjoin(savedir,'deep_plasma'))==False:

    if loadModel:
        print('Model data exists. loading model ...')
        deep_approx = keras.models.load_model(pjoin(savedir,'deep_plasma'))
    else:
        print('Model data does not exist. Building model ...')
        # Build model
        deep_approx = keras.models.Sequential()
        deep_approx.add(layers.Conv2D(64,2,2, input_shape=(2,64,32),activation='relu'))
        # deep_approx.add(layers.Conv2D(32,2,2,activation='relu'))
        # deep_approx.add(layers.Conv2D(16,2,2,activation='relu'))
        # print(deep_approx.output_shape)
        deep_approx.add(layers.Flatten())
        deep_approx.add(layers.Dense(64,activation='relu'))
        deep_approx.add(layers.Dense(1, activation='softmax'))
        # deep_approx.add(layers.Flatten())
        # deep_approx.add(layers.Dense(50, activation='elu'))
        # deep_approx.add(layers.Dense(50, activation='elu'))
        # deep_approx.add(layers.Dense(1, activation='linear'))
        print(deep_approx.output_shape)
        # Compile model
        deep_approx.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #FIT
        validation_split = 0.2
        verbosity = 1
        history = deep_approx.fit(X_train, y_train,
                    epochs=10, batch_size=32,
                    verbose=verbosity,
                    validation_split=validation_split)

        deep_approx.summary()
        deep_approx.save(pjoin(savedir,'deep_plasma'))


        # history.history contains loss information
        if histplot:
            idx0 = 1
            plt.figure()
            plt.semilogy(history.history['loss'][idx0:], '.-', lw=2)
            plt.semilogy(history.history['val_loss'][idx0:], '.-', lw=2)
            plt.xlabel('epochs')
            plt.ylabel('Validation loss')
            plt.legend(['training loss', 'validation loss'])
            plt.show()
    return deep_approx
