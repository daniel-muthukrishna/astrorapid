import os
import numpy as np
import pickle
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.layers import LSTM, GRU
from tensorflow.python.keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed, Masking
from tensorflow.python.keras.layers.convolutional import Conv1D, Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling1D, MaxPooling2D
import matplotlib.pyplot as plt


def train_model(X_train, X_test, y_train, y_test, sample_weights=None, fig_dir='.', retrain=True, epochs=25, plot_loss=True):
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
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=500,
                            verbose=2, sample_weight=sample_weights)

        print(model.summary())
        model.save(model_filename)

        with open(os.path.join(fig_dir, "model_history.pickle"), 'wb') as fp:
            pickle.dump(history.history, fp)

        if plot_loss:
            plot_history(history.history, fig_dir)

    return model


def plot_history(history, fig_dir):
    # Plot loss vs epochs
    plt.figure(figsize=(12,10))
    train_loss = history['loss']
    val_loss = history['val_loss']
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title(os.path.basename(fig_dir).replace('_', '-'))
    plt.savefig(os.path.join(fig_dir, "model_loss_history.pdf"))

    # Plot accuracy vs figure
    plt.figure(figsize=(12,10))
    if 'accuracy' in history:
        train_acc = history['accuracy']
        val_acc = history['val_accuracy']
    else:
        train_acc = history['acc']
        val_acc = history['val_acc']
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title(os.path.basename(fig_dir).replace('_', '-'))
    plt.savefig(os.path.join(fig_dir, "model_accuracy_history.pdf"))
