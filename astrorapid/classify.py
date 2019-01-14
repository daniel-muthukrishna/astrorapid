import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from process_light_curves import read_multiple_light_curves, prepare_input_arrays

plt.rcParams['text.usetex'] = True

plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']


COLCLASS = {'Pre-explosion': 'grey', 'SNIa-norm': 'tab:green', 'SNIbc': 'tab:orange', 'SNII': 'tab:blue',
            'SNIa-91bg': 'tab:red', 'SNIa-x': 'tab:purple', 'Class A': 'tab:brown', 'Kilonova': '#aaffc3',
            'SLSN-I': 'tab:olive', 'Class B': 'tab:cyan', 'Class C': '#FF1493', 'Class D': 'navy', 'TDE': 'tab:pink'}
COLPB = {'u': 'tab:blue', 'g': 'tab:blue', 'r': 'tab:orange', 'i': 'm', 'z': 'k', 'Y': 'y'}
MARKPB = {'g': 'o', 'r': 's'}
ALPHAPB = {'g': 0.3, 'r': 1.}


def classify(X, model_filepath):
    model = load_model(model_filepath)

    y_predict = model.predict(X)

    return y_predict


def main(model_filepath='keras_model.hdf5'):
    passbands = ('g', 'r')
    light_curve_list = [(mjd, flux, fluxerr, passband, zeropoint, photflag, ra, dec, objid, redshift, mwebv)]
    lightcurves = read_multiple_light_curves(light_curve_list)
    prepare_input_arrays(lightcurves, passbands)


if __name__ == '__main__':
    main()
