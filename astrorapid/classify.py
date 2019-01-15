import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict
from keras.models import load_model
import matplotlib
import matplotlib.animation as animation

from astrorapid.process_light_curves import read_multiple_light_curves
from astrorapid.prepare_arrays import PrepareInputArrays

plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

CLASS_NAMES = ['Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'Class A', 'Kilonova', 'SLSN-I',
               'Class B', 'Class C', 'Class D', 'TDE']
CLASS_COLOR = {'Pre-explosion': 'grey', 'SNIa-norm': 'tab:green', 'SNIbc': 'tab:orange', 'SNII': 'tab:blue',
               'SNIa-91bg': 'tab:red', 'SNIa-x': 'tab:purple', 'Class A': 'tab:brown', 'Kilonova': '#aaffc3',
               'SLSN-I': 'tab:olive', 'Class B': 'tab:cyan', 'Class C': '#FF1493', 'Class D': 'navy', 'TDE': 'tab:pink'}
PB_COLOR = {'u': 'tab:blue', 'g': 'tab:blue', 'r': 'tab:orange', 'i': 'm', 'z': 'k', 'Y': 'y'}
PB_MARKER = {'g': 'o', 'r': 's'}
PB_ALPHA = {'g': 0.3, 'r': 1.}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class Classify(object):
    def __init__(self, light_curves, known_redshift=True, model_filepath='', passbands=('g', 'r')):
        """ Takes a list of photometric information and classifies light curves as a function of time

        Parameters
        ----------
        light_curves : list
            Is a list of tuples. Each tuple contains the light curve information of a transient object in the form
            (mjd, flux, fluxerr, passband, zeropoint, photflag, ra, dec, objid, redshift, mwebv).
            Here, mjd, flux, fluxerr, passband, zeropoint, and photflag are arrays.
            ra, dec, objid, redshift, and mwebv are floats
        known_redshift : bool
            Different model to be used if redshift is not known.
        model_filepath : str
            Optional argument. The model is taken from the pre-trained model ZTF model if not specified.
        passbands : tuple
            Optional argument. A tuple listing each passband. E.g. ('g', 'r').

        """
        self.light_curves = light_curves
        self.known_redshift = known_redshift
        self.passbands = passbands

        if self.known_redshift:
            self.model_filepath = os.path.join(SCRIPT_DIR, 'keras_model_with_redshift.hdf5')
            self.contextual_info = (0,)
        else:
            self.model_filepath = os.path.join(SCRIPT_DIR, 'keras_model_no_redshift.hdf5')
            self.contextual_info = ()

        if model_filepath != '' and os.path.exists(model_filepath):
            self.model_filepath = model_filepath
            print("Invalid keras model. Using default model...")

        self.model = load_model(self.model_filepath)

    def process_light_curves(self):
        processed_lightcurves = read_multiple_light_curves(self.light_curves, known_redshift=self.known_redshift, training_set_parameters=None)
        prepareinputarrays = PrepareInputArrays(self.passbands, self.contextual_info)
        X = prepareinputarrays.prepare_input_arrays(processed_lightcurves)

        return X

    def get_predictions(self):
        self.X, self.orig_lc, self.timesX, self.objids = self.process_light_curves()

        self.y_predict = self.model.predict(self.X)

        return self.y_predict

    def plot_light_curves_and_classifications(self, indexes_to_plot=None):
        """
        Plot light curve (top panel) and classifications (bottom panel) vs time.

        Parameters
        ----------
        indexes_to_plot : tuple
            The indexes listed in the tuple will be plotted according to the order of the input light curves.
            E.g. (0, 1, 3, 5) will plot the zeroth, first, third and fifth light curves and classifications.
            If None or True, then all light curves will be plotted

        """

        font = {'family': 'normal',
                'size': 33}
        matplotlib.rc('font', **font)

        if indexes_to_plot is None or indexes_to_plot is True:
            indexes_to_plot = np.arange(len(self.y_predict))

        if not hasattr(self, 'y_predict'):
            self.get_predictions()

        for idx in indexes_to_plot:
            argmax = self.timesX[idx].argmax() + 1
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15),
                                           num="classification_vs_time_{}".format(idx), sharex=True)
            ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
            ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)

            for pb in self.passbands:
                if pb in self.orig_lc[idx].keys():
                    ax1.errorbar(self.orig_lc[idx][pb]['time'], self.orig_lc[idx][pb]['fluxRenorm'],
                                 yerr=self.orig_lc[idx][pb]['fluxErrRenorm'], fmt=PB_MARKER[pb], label=pb,
                                 c=PB_COLOR[pb], lw=3, markersize=10)

            for classnum, classname in enumerate(CLASS_NAMES):
                ax2.plot(self.timesX[idx][:argmax], self.y_predict[idx][:, classnum][:argmax], '-', label=classname,
                         color=CLASS_COLOR[classname], linewidth=3)
            ax1.legend(frameon=False, fontsize=33)
            ax2.legend(frameon=False, fontsize=23.5)  # , ncol=2)
            ax2.set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
            ax1.set_ylabel("Relative Flux")  # , fontsize=15)
            ax2.set_ylabel("Class Probability")  # , fontsize=18)
            ax1.set_ylim(-0.1, 1.1)
            ax2.set_ylim(0, 1)
            ax1.set_xlim(-70, 80)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added
            plt.tight_layout()
            fig.subplots_adjust(hspace=0)
            plt.savefig('classification_vs_time_{}.pdf'.format(self.objids[idx]))
            plt.close()

    def plot_classification_animation(self, indexes_to_plot=None):
        """ Plot light curve (top panel) and classifications (bottom panel) vs time as an mp4 animation.

        Parameters
        ----------
        indexes_to_plot : tuple
            The indexes listed in the tuple will be plotted according to the order of the input light curves.
            E.g. (0, 1, 3, 5) will plot the zeroth, first, third and fifth light curves and classifications.
            If None or True, then all light curves will be plotted

        """

        font = {'family': 'normal',
                'size': 33}
        matplotlib.rc('font', **font)

        if indexes_to_plot is None or indexes_to_plot is True:
            indexes_to_plot = np.arange(len(self.y_predict))

        if not hasattr(self, 'y_predict'):
            self.get_predictions()

        for idx in indexes_to_plot:
            argmax = self.timesX[idx].argmax() + 1
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15),
                                           num="animation_classification_vs_time_{}".format(idx), sharex=True)

            ax1.legend(frameon=False, fontsize=33)
            ax2.legend(frameon=False, fontsize=23.5)  # , ncol=2)
            ax2.set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
            ax1.set_ylabel("Relative Flux")  # , fontsize=15)
            ax2.set_ylabel("Class Probability")  # , fontsize=18)
            ax1.set_ylim(-0.1, 1.1)
            ax2.set_ylim(0, 1)
            ax1.set_xlim(-70, 80)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added
            plt.tight_layout()
            fig.subplots_adjust(hspace=0)
            ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
            ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, bitrate=1800)

            def animate(i):
                for pbidx, pb in enumerate(self.passbands):
                    ax1.plot(self.timesX[idx][:argmax][:int(i + 1)], self.X[idx][:, pbidx][:argmax][:int(i + 1)],
                             label=pb, c=PB_COLOR[pb], lw=3)  # , markersize=10, marker=MARKPB[pb])

                for classnum, classname in enumerate(CLASS_NAMES):
                    ax2.plot(self.timesX[idx][:argmax][:int(i + 1)], self.y_predict[idx][:, classnum][:argmax][:int(i + 1)],
                             '-', label=classname, color=CLASS_COLOR[classname], linewidth=3)

                # Don't repeat legend items
                ax1.legend(frameon=False, fontsize=33)
                ax2.legend(frameon=False, fontsize=23.5)
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                by_label1 = OrderedDict(zip(labels1, handles1))
                by_label2 = OrderedDict(zip(labels2, handles2))
                ax1.legend(by_label1.values(), by_label1.keys(), frameon=False, fontsize=33)
                ax2.legend(by_label2.values(), by_label2.keys(), frameon=False, fontsize=23.5)

            ani = animation.FuncAnimation(fig, animate, frames=50, repeat=True)
            ani.save(os.path.join('classification_vs_time_{}.mp4'.format(self.objids[idx])), writer=writer)
