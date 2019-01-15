import os
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Input
from keras.layers import LSTM, GRU
from keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D

from astrorapid.prepare_arrays import PrepareTrainingSetArrays


def train_model(X_train, X_test, y_train, y_test, sample_weights=None, fig_dir='.', retrain=True):
    model_filename = os.path.join(fig_dir, "keras_model.hdf5")

    if not retrain and os.path.isfile(model_filename):
        model = load_model(model_filename)
    else:
        num_classes = y_test.shape[-1]

        model = Sequential()

        # model.add(Conv1D(filters=32, kernel_size=3))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling1D(pool_size=1))
        # model.add(Dropout(0.2, seed=42))

        model.add(GRU(100, return_sequences=True))
        model.add(Dropout(0.2, seed=42))
        model.add(BatchNormalization())

        model.add(GRU(100, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2, seed=42))

        model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64, verbose=2, sample_weight=sample_weights)

        print(model.summary())
        model.save(model_filename)

    return model


def main():
    passbands = ('g', 'r')
    contextual_info = ()

    aggregate_classes = True
    reread_hdf5_data = False
    retrain_rnn = False

    otherchange = ''
    nchunks = 1000

    # Train + Test cuts
    zcut = 0.5
    bcut = True
    variablescut = True

    training_set_dir = 'training_set_files'
    if not os.path.exists(training_set_dir):
        os.makedirs(training_set_dir)


    fpath = '/Users/danmuth/PycharmProjects/earlyclass/lc_data_MSIP_ZTF_20180716_full_lc_z0.5_mr23_ebv02_widercuts5.hdf5'

    fig_dir = os.path.join(training_set_dir, 'Figures', 'classify', 'ZTF_25epochs{}ag{}_ci{}_fp{}_zcut{}_bcut{}_varcut{}'.format(otherchange, aggregate_classes, contextual_info, os.path.basename(fpath), zcut, bcut, variablescut))
    for dirname in [fig_dir, fig_dir+'/cf_since_trigger', fig_dir+'/cf_since_t0', fig_dir+'/roc_since_trigger', fig_dir+'/lc_pred', fig_dir+'/pr_since_trigger', fig_dir+'/truth_table_since_trigger']:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    preparearrays = PrepareTrainingSetArrays(passbands, contextual_info, reread_hdf5_data, aggregate_classes, bcut, zcut, variablescut, nchunks=10000)
    X_train, X_test, y_train, y_test, labels_train, labels_test, class_names, class_weights, sample_weights, timesX_train, timesX_test, orig_lc_train, orig_lc_test, objids_train, objids_test = preparearrays.prepare_training_set_arrays(fpath, otherchange)
    model = train_model(X_train, X_test, y_train, y_test, sample_weights=sample_weights, fig_dir=fig_dir, retrain=retrain_rnn)
    plot_metrics(class_names, model, X_test, y_test, fig_dir, timesX_test=timesX_test, orig_lc_test=orig_lc_test, objids_test=objids_test, passbands=passbands)


if __name__ == '__main__':
    main()
