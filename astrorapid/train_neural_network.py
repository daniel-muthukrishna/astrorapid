import os
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.layers import LSTM, GRU
from tensorflow.python.keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed, Masking
from tensorflow.python.keras.layers.convolutional import Conv1D, Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling1D, MaxPooling2D


def train_model(X_train, X_test, y_train, y_test, sample_weights=None, fig_dir='.', retrain=True, epochs=25):
    """ Train Neural Network classifier and save model. """

    model_filename = os.path.join(fig_dir, "keras_model.hdf5")
    #TODO: Try standard scaling and try normalising by peak and try reinputting such that it always normalises by largest value so far

    # colour = np.log10(X_train[:,:,0]) - np.log10(X_train[:,:,1])
    # X_train = np.dstack((X_train, colour))
    # colour = np.log10(X_test[:,:,0]) - np.log10(X_test[:,:,1])
    # X_test = np.dstack((X_test, colour))
    print("training...")
    if not retrain and os.path.isfile(model_filename):
        model = load_model(model_filename)
    else:
        num_classes = y_test.shape[-1]

        model = Sequential()

        model.add(Masking(mask_value=0.))

        # model.add(Conv1D(filters=32, kernel_size=3))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling1D(pool_size=1))
        # model.add(Dropout(0.2, seed=42))

        model.add(LSTM(100, return_sequences=True))
        # model.add(Dropout(0.2, seed=42))
        # model.add(BatchNormalization())

        model.add(LSTM(100, return_sequences=True))
        # model.add(Dropout(0.2, seed=42))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.2, seed=42))

        model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=500, verbose=2, sample_weight=sample_weights)

        print(model.summary())
        model.save(model_filename)

    return model
