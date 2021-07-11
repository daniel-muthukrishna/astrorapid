import os
import numbers
import numpy as np
from collections import OrderedDict
from tensorflow.keras.models import load_model
from pkg_resources import resource_filename
from distutils.spawn import find_executable
from tcn import TCN, tcn_full_summary

from astrorapid.process_light_curves import read_multiple_light_curves
from astrorapid.prepare_input import PrepareInputArrays

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import matplotlib
    import matplotlib.animation as animation

    # Check if latex is installed
    if find_executable('latex'):
        plt.rcParams['text.usetex'] = True
    plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

except ImportError:
    print("Warning: You will need to install 'matplotlib' if you wish to plot the classifications.")


CLASS_COLOR = {'Pre-explosion': 'grey', 'SNIa-norm': 'tab:green', 'SNIbc': 'tab:orange', 'SNII': 'tab:blue',
               'SNIa-91bg': 'tab:red', 'SNIa-x': 'tab:purple', 'point-Ia': 'tab:brown', 'Kilonova': '#aaffc3',
               'SLSN-I': 'tab:olive', 'PISN': 'tab:cyan', 'ILOT': '#FF1493', 'CART': 'navy', 'TDE': 'tab:pink',
               'AGN': 'bisque', 'Ia': 'tab:green', 'SLSN': 'tab:olive', 'II': 'tab:blue', 'IIn': 'tab:brown',
               'Ibc': 'tab:orange', 'CC': 'blue'}
PB_COLOR = {'u': 'tab:blue', 'g': 'tab:blue', 'r': 'tab:orange', 'i': 'm', 'z': 'k', 'Y': 'y'}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class Classify(object):
    def __init__(self, model_name='ZTF_known_redshift', model_filepath='', known_redshift=True, passbands=('g', 'r'),
                 class_names=('Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'Kilonova', 'SLSN-I', 'TDE'),
                 nobs=50, mintime=-70, maxtime=80, timestep=3.0, bcut=False, zcut=None, graph=None, model=None):
        """ Takes a list of photometric information and classifies light curves as a function of time

        Parameters
        ----------
        model_name : str
            The name of the pretrained model to use.
            Choose one of: 'PS1_known_redshift_Ia_CC_SLSN', 'PS1_known_redshift_Ia_II_IIn_Ibc_SLSN', 'ZTF_known_redshift', 'ZTF_unknown_redshift'.
        model_filepath : str
            Optional argument. If not specified, the pre-trained model specified in the model_name argument is used.
        known_redshift : bool
            Optional argument. Only needed if you are not using a default model and have specified model_filepath.
        passbands : tuple
            Optional argument. Only needed if you are not using a default model and have specified model_filepath.
            A tuple listing each passband. E.g. ('g', 'r').
        class_names : tuple
            Optional argument. Only needed if you are not using a default model and have specified model_filepath.
            List of class names that the model has been trained on. Note that this must be in the same order
            as used in training for the model specified in the argument model_filepath.
            If you are using the default model, leave this argument out.
        nobs : int
            Optional argument. Only needed if you are not using a default model and have specified model_filepath.
            Number of points to use in interpolation of light curve between mintime and maxtime.
            Do not change this argument unless you are using your own model_filepath
            and have changed this during training.
        mintime : int
            Optional argument. Only needed if you are not using a default model and have specified model_filepath.
            Days from trigger (minimum) to extract from light curve.
        maxtime : int
            Optional argument. Only needed if you are not using a default model and have specified model_filepath.
            Days from trigger (maximum) to extract from light curve.
        timestep : float
            Optional argument. Only needed if you are not using a default model and have specified model_filepath.
            Time-step between interpolated points in light curve.
            Do not change this argument unless you are using your own model_filepath
            and have changed this during training.
        bcut : bool
            Cut on galactic latitude.
            Do not set unless you know what you are doing.
        zcut : float or None
            Remove redshifts above this value.
            Do not set unless you know what you are doing.
        graph : tensorflow graph
            Do not set unless you know what you are doing.
            If you are running astrorapid in multiple threads you may need to predefine this
            This would have been created with Tensorflow i.e. graph = tf.get_default_graph()
        model : tensorflow model
            Do not set unless you know what you are doing.
            If you are running astrorapid in multiple threads you may need to predefine this
            This would have been created with keras' load_model function e.g. model = load_model('keras_model.hdf5')

        """

        self.bcut = bcut
        self.zcut = zcut

        if model_filepath != '' and os.path.exists(model_filepath):
            self.model_filepath = model_filepath
            self.contextual_info = ['redshift',] if known_redshift else []
            self.known_redshift = known_redshift
            self.passbands = passbands
            self.class_names = class_names
            self.nobs = nobs
            self.mintime = mintime
            self.maxtime = maxtime
            self.timestep = timestep
        else:
            if model_name == 'ZTF_known_redshift':
                self.model_filepath = os.path.join(SCRIPT_DIR, 'ZTF_known_redshift.hdf5')
                self.contextual_info = ['redshift',]
                self.known_redshift = True
                self.passbands=('g', 'r')
                self.class_names = ('Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'Kilonova', 'SLSN-I', 'TDE')
                self.nobs = 50
                self.mintime = -70
                self.maxtime = 80
                self.timestep = 3.0
            elif model_name == 'ZTF_unknown_redshift':
                self.model_filepath = os.path.join(SCRIPT_DIR, 'ZTF_unknown_redshift.hdf5')
                self.contextual_info = []
                self.known_redshift = False
                self.passbands = ('g', 'r')
                self.class_names = ('Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'Kilonova', 'SLSN-I', 'TDE')
                self.nobs = 50
                self.mintime = -70
                self.maxtime = 80
                self.timestep = 3.0
            elif model_name == 'PS1_known_redshift_Ia_CC_SLSN':
                self.model_filepath = os.path.join(SCRIPT_DIR, 'PS1_known_redshift_Ia_CC_SLSN.hdf5')
                self.contextual_info = ['redshift',]
                self.known_redshift = True
                self.passbands = ('g', 'r', 'i', 'z')
                self.class_names = ('Pre-explosion', 'SLSN', 'CC', 'Ia')
                self.nobs = 50
                self.mintime = -70
                self.maxtime = 80
                self.timestep = 3.0
            elif model_name == 'PS1_known_redshift_Ia_II_IIn_Ibc_SLSN':
                self.model_filepath = os.path.join(SCRIPT_DIR, 'PS1_known_redshift_Ia_II_IIn_Ibc_SLSN.hdf5')
                self.contextual_info = ['redshift',]
                self.known_redshift = True
                self.passbands = ('g', 'r', 'i', 'z')
                self.class_names = ('Pre-explosion', 'SLSN', 'II', 'IIn', 'Ia', 'Ibc')
                self.nobs = 50
                self.mintime = -70
                self.maxtime = 80
                self.timestep = 3.0
            elif model_name == 'PS1_unknown_redshift_Ia_CC_SLSN':
                self.model_filepath = os.path.join(SCRIPT_DIR, 'PS1_unknown_redshift_Ia_CC_SLSN.hdf5')
                self.contextual_info = []
                self.known_redshift = False
                self.passbands = ('g', 'r', 'i', 'z')
                self.class_names = ('Pre-explosion', 'SLSN', 'II', 'IIn', 'Ia', 'Ibc')
                self.nobs = 50
                self.mintime = -70
                self.maxtime = 80
                self.timestep = 3.0
            else:
                raise Exception("Invalid model_name. Select a valid model_name from the following "
                                "('PS1_known_redshift', 'PS1_unknown_redshift', "
                                "'ZTF_known_redshift', 'ZTF_unknown_redshift'), "
                                "or set model_filepath to a valid path to your own trained model.")

        print(self.model_filepath)
        self.graph = graph
        if graph is not None and model is not None:
            self.model = model
        else:
            self.model = load_model(self.model_filepath, custom_objects={'TCN': TCN})

    def process_light_curves(self, light_curves, other_meta_data=None):
        """

        Parameters
        ----------
        light_curves: list of tuples
            Each tuple in the list is of the form: (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv)
            for each transient.
        other_meta_data: list of dictionaries or None
            Each dictionary in the list contains any additional meta data to be used as contextual_info
            when classifying (if the classifier being used was trained on the specified contextual_info).
            E.g. other_meta_data = [{'hosttype': 3, 'host_dist': 200}, {'hosttype': 2, 'host_dist': 150},]


        Returns
        -------

        """
        processed_lightcurves = read_multiple_light_curves(light_curves, known_redshift=self.known_redshift,
                                                           training_set_parameters=None,
                                                           other_meta_data=other_meta_data)
        prepareinputarrays = PrepareInputArrays(self.passbands, self.contextual_info, self.bcut, self.zcut,
                                                self.nobs, self.mintime, self.maxtime, self.timestep)
        X, orig_lc, timesX, objids_list, trigger_mjds = prepareinputarrays.prepare_input_arrays(processed_lightcurves)

        # # REMOVE CORRECTION FACTOR IF NOT USED
        # correction_factor = np.load('astrorapid/correction_factor.npy')
        # for i, pb in enumerate(self.passbands):
        #     X[:, :, i] = X[:, :, i] / correction_factor[i]

        return X, orig_lc, timesX, objids_list, trigger_mjds

    def _do_error_checks(self, light_curves):
        assert isinstance(light_curves, (list, np.ndarray))
        for light_curve in light_curves:
            assert len(light_curve) == 10
            mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv = light_curve
            _lcshape = len(mjd)
            assert _lcshape == len(flux)
            assert _lcshape == len(fluxerr)
            assert _lcshape == len(passband)
            assert _lcshape == len(photflag)
            assert all(isinstance(n, numbers.Number) and np.isfinite(n) for n in mjd)
            assert all(isinstance(n, numbers.Number) and np.isfinite(n) for n in flux)
            assert all(isinstance(n, numbers.Number) and np.isfinite(n) for n in fluxerr)
            assert all(isinstance(n, numbers.Number) and np.isfinite(n) for n in photflag)
            assert all(isinstance(n, str) for n in passband)
            assert isinstance(ra, numbers.Number)
            assert isinstance(dec, numbers.Number)
            assert isinstance(mwebv, numbers.Number)
            assert np.isfinite(ra)
            assert np.isfinite(dec)
            assert np.isfinite(mwebv)
            if self.known_redshift:
                assert isinstance(redshift, numbers.Number)
                assert np.isfinite(redshift)

    def get_predictions(self, light_curves, other_meta_data=None, return_predictions_at_obstime=False, return_objids=False):
        """ Return the classification accuracies as a function of time for each class

        Parameters
        ----------
        light_curves : list
            Is a list of tuples. Each tuple contains the light curve information of a transient object in the form
            (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv).
            Here, mjd, flux, fluxerr, passband, and photflag are arrays.
            And ra, dec, objid, redshift, and mwebv are floats.
                mjd: list of Modified Julian Dates of light curve

                flux: list of fluxes at each mjd

                fluxerr: list of flux errors

                passband: list of strings indicating the passband. 'r' or 'g' for r-band or g-band observations.

                photflag: list of flags identifying whether the observation is a detection (4096), non-detection (0),
                or the first detection (6144).

                ra: Right Ascension (float value).

                dec: Declination (float value).

                objid: Object Identifier (String).

                redshift: Cosmological redshift of object (float). Set to NoneType if redshift is unknown.

                mwebv: Milky way extinction.

        other_meta_data: list of dictionaries or None
            Each dictionary in the list contains any additional meta data to be used as contextual_info
            when classifying (if the classifier being used was trained on the specified contextual_info).
            E.g. other_meta_data = [{'hosttype': 3, 'host_dist': 200}, {'hosttype': 2, 'host_dist': 150},]

        return_predictions_at_obstime: bool
            Return the predictions at the observation times instead of at the 50 interpolated timesteps.
        return_objids : bool, optional
            If True, also return the object IDs (objids) in the same order as the returned predictions.

        Returns
        -------
        y_predict: array
            Classification probability vector at each time step for each object.
            Array of shape (s, n, m) is returned.
            Where s is the number of obejcts that are classified,
            n is the number of times steps, and m is the number of classes.
        time_steps: array
            MJD time steps corresponding to the timesteps of the y_predict array.
        objids : array, optional
            The object ids (objids) that were input into light_curves are returned in the same order as y_predict.
            Only provided if return_objids is True.
        """

        self._do_error_checks(light_curves)

        self.X, self.orig_lc, self.timesX, self.objids, self.trigger_mjds = self.process_light_curves(light_curves,
                                                                                                      other_meta_data=other_meta_data)
        nobjects = len(self.objids)

        if nobjects == 0:
            print("No objects to classify. These may have been removed from the chosen selection cuts")
            if return_objids:
                return None, None, self.objids
            else:
                return None, None

        if self.graph is not None:
            with self.graph.as_default():
                self.y_predict = self.model.predict(self.X)
        else:
            self.y_predict = self.model.predict(self.X)

        argmax = self.timesX.argmax(axis=1) + 1

        if return_predictions_at_obstime:
            (s, n, m) = self.y_predict.shape  # (s, n, m) = (num light curves, num timesteps, num classes)
            y_predict = []
            time_steps = []
            for idx in range(s):
                obs_time = []
                lc_data = self.orig_lc[idx]
                used_passbands = [pb for pb in self.passbands if pb in lc_data['passband']]
                for pb in used_passbands:
                    pbmask = lc_data['passband'] == pb
                    if pb in self.orig_lc[idx]:
                        obs_time.append(lc_data[pbmask]['time'].data)
                obs_time = np.array(obs_time)
                obs_time = np.sort(obs_time[~np.isnan(obs_time)])
                y_predict_at_obstime = []
                for classnum, classname in enumerate(self.class_names):
                    y_predict_at_obstime.append(np.interp(obs_time, self.timesX[idx][:argmax[idx]], self.y_predict[idx][:, classnum][:argmax[idx]]))
                y_predict.append(np.array(y_predict_at_obstime).T)
                time_steps.append(obs_time + self.trigger_mjds[idx])
        else:
            y_predict = [self.y_predict[i][:argmax[i]] for i in range(nobjects)]
            time_steps = [self.timesX[i][:argmax[i]] + self.trigger_mjds[i] for i in range(nobjects)]

        if return_objids:
            return y_predict, time_steps, self.objids

        return y_predict, time_steps

    def plot_light_curves_and_classifications(self, indexes_to_plot=None, step=True, use_interp_flux=False, figdir='.',
                                              plot_matrix_input=False, light_curves=None, show_plot=False):
        """
        Plot light curve (top panel) and classifications (bottom panel) vs time.

        Parameters
        ----------
        indexes_to_plot : tuple
            The indexes listed in the tuple will be plotted according to the order of the input light curves.
            E.g. (0, 1, 3, 5) will plot the zeroth, first, third and fifth light curves and classifications.
            If None or True, then all light curves will be plotted
        step : bool
            Plot step function along data points instead of interpolating classifications between data.
        use_interp_flux : bool
            Use all 50 timesteps when plotting classification probabilities rather than just at the timesteps with data.
        figdir : str
            Directory to save figure.
        plot_matrix_input : bool
            Plots the interpolated light curve passed into the neural network on top of the observations.
        light_curves : list
            This argument is only required if the get_predictions() method has not been run.
            Is a list of tuples. Each tuple contains the light curve information of a transient object in the form
            (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv).
            Here, mjd, flux, fluxerr, passband, and photflag are arrays.
            ra, dec, objid, redshift, and mwebv are floats
        show_plot : bool
            Run plt.show() after each object plotted.

        """

        font = {'family': 'normal',
                'size': 33}
        matplotlib.rc('font', **font)

        if indexes_to_plot is None or indexes_to_plot is True:
            indexes_to_plot = np.arange(len(self.y_predict))

        if not hasattr(self, 'y_predict'):
            self.get_predictions(light_curves)

        if plot_matrix_input:
            alpha_observations = 0.2
        else:
            alpha_observations = 1.
        for idx in indexes_to_plot:
            argmax = self.timesX[idx].argmax() + 1
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15),
                                           num="classification_vs_time_{}".format(idx), sharex=True)
            # ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
            # ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)

            lc_data = self.orig_lc[idx]
            used_passbands = [pb for pb in self.passbands if pb in lc_data['passband']]

            for pbidx, pb in enumerate(used_passbands):
                pbmask = lc_data['passband'] == pb
                ax1.errorbar(lc_data[pbmask]['time'], lc_data[pbmask]['flux'],
                             yerr=lc_data[pbmask]['fluxErr'], fmt='o', label=pb,
                             c=PB_COLOR[pb], lw=3, markersize=10, alpha=alpha_observations)
                if plot_matrix_input:
                    ax1.plot(self.timesX[idx][:argmax], self.X[idx][:, pbidx][:argmax], c=PB_COLOR[pb], lw=3)

            new_t = np.concatenate([lc_data[lc_data['passband'] == pb]['time'].data for pb in used_passbands])
            new_t = np.sort(new_t[~np.isnan(new_t)])
            if not use_interp_flux and not plot_matrix_input:
                new_y_predict = []
                for classnum, classname in enumerate(self.class_names):
                    new_y_predict.append(np.interp(new_t, self.timesX[idx][:argmax], self.y_predict[idx][:, classnum][:argmax]))

            for classnum, classname in enumerate(self.class_names):
                if use_interp_flux or plot_matrix_input:
                    ax2.plot(self.timesX[idx][:argmax], self.y_predict[idx][:, classnum][:argmax], '-', label=classname,
                             color=CLASS_COLOR[classname], linewidth=3)
                else:
                    if step:
                        ax2.step(new_t, new_y_predict[classnum], '-', label=classname, color=CLASS_COLOR[classname], linewidth=3, where='post')
                    else:
                        ax2.plot(new_t, new_y_predict[classnum], '-', label=classname, color=CLASS_COLOR[classname], linewidth=3,)

            ax1.legend(frameon=True, fontsize=33)#, loc='lower right')
            ax2.legend(frameon=True, fontsize=21.5)#, loc='center right')  # , ncol=2)
            ax2.set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
            ax1.set_ylabel("Relative Flux")  # , fontsize=15)
            ax2.set_ylabel("Class Probability")  # , fontsize=18)
            # ax1.set_ylim(-0.1, 1.1)
            ax2.set_ylim(0, 1)
            ax1.set_xlim(left=min(new_t))  # ax1.set_xlim(-70, 80)
            # ax1.grid(True)
            # ax2.grid(True)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added
            plt.tight_layout()
            fig.subplots_adjust(hspace=0)
            savename = 'classification_vs_time_{}{}{}.pdf'.format(self.objids[idx], '_step' if step else '', '_no_interp' if not use_interp_flux else '')
            plt.savefig(os.path.join(figdir, savename))
            # plt.savefig("{}.png".format(savename))
            if show_plot:
                plt.show()
            plt.close()

        return self.orig_lc, self.timesX, self.y_predict

    def plot_classification_animation(self, indexes_to_plot=None, figdir='.', light_curves=None):
        """ Plot light curve (top panel) and classifications (bottom panel) vs time as an mp4 animation.

        Parameters
        ----------
        indexes_to_plot : tuple
            The indexes listed in the tuple will be plotted according to the order of the input light curves.
            E.g. (0, 1, 3, 5) will plot the zeroth, first, third and fifth light curves and classifications.
            If None or True, then all light curves will be plotted
        figdir : str
            Directory to save figure.
        light_curves : list
            This argument is only required if the get_predictions() method has not been run.
            Is a list of tuples. Each tuple contains the light curve information of a transient object in the form
            (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv).
            Here, mjd, flux, fluxerr, passband, and photflag are arrays.
            ra, dec, objid, redshift, and mwebv are floats

        """

        font = {'family': 'normal',
                'size': 33}
        matplotlib.rc('font', **font)

        if indexes_to_plot is None or indexes_to_plot is True:
            indexes_to_plot = np.arange(len(self.y_predict))

        if not hasattr(self, 'y_predict'):
            self.get_predictions(light_curves)

        for idx in indexes_to_plot:
            lc_data = self.orig_lc[idx]
            used_passbands = [pb for pb in self.passbands if pb in lc_data['passband']]

            new_t = np.concatenate([lc_data[lc_data['passband'] == pb]['time'].data for pb in used_passbands])
            all_flux = list(lc_data[lc_data['passband'] == 'g']['flux']) + list(lc_data[lc_data['passband'] == 'r']['flux'])

            argmax = self.timesX[idx].argmax() + 1
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15),
                                           num="animation_classification_vs_time_{}".format(idx), sharex=True)

            ax1.legend(frameon=False, fontsize=33)
            ax2.legend(frameon=False, fontsize=23.5)  # , ncol=2)
            ax2.set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
            ax1.set_ylabel("Relative Flux")  # , fontsize=15)
            ax2.set_ylabel("Class Probability")  # , fontsize=18)
            ax1.set_ylim(min(all_flux), max(all_flux))
            ax2.set_ylim(0, 1)
            ax1.set_xlim(min(new_t), max(new_t))  # ax1.set_xlim(-70, 80)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added
            plt.tight_layout()
            fig.subplots_adjust(hspace=0)
            ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
            ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, bitrate=1800)

            def animate(i):
                for pbidx, pb in enumerate(used_passbands):
                    ax1.plot(self.timesX[idx][:argmax][:int(i + 1)], self.X[idx][:, pbidx][:argmax][:int(i + 1)],
                             label=pb, c=PB_COLOR[pb], lw=3)  # , markersize=10, marker='.')

                for classnum, classname in enumerate(self.class_names):
                    ax2.plot(self.timesX[idx][:argmax][:int(i + 1)],
                             self.y_predict[idx][:, classnum][:argmax][:int(i + 1)],
                             '-', label=classname, color=CLASS_COLOR[classname], linewidth=3)

                # Don't repeat legend items
                ax1.legend(frameon=False, fontsize=33)
                ax2.legend(frameon=False, fontsize=23.5)
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                by_label1 = OrderedDict(zip(labels1, handles1))
                by_label2 = OrderedDict(zip(labels2, handles2))
                ax1.legend(by_label1.values(), by_label1.keys(), frameon=False, fontsize=33, loc='lower right')
                ax2.legend(by_label2.values(), by_label2.keys(), frameon=False, fontsize=21.5, loc='center right')

            ani = animation.FuncAnimation(fig, animate, frames=50, repeat=True)
            savename = os.path.join(figdir,'classification_vs_time_{}.mp4'.format(self.objids[idx]))
            ani.save(savename, writer=writer)

    def plot_classification_animation_step(self, indexes_to_plot=None, figdir='.', light_curves=None):
        """
        Plot light curve (top panel) and classifications (bottom panel) vs time as an mp4 animation
        as step function.

        Parameters
        ----------
        indexes_to_plot : tuple
            The indexes listed in the tuple will be plotted according to the order of the input light curves.
            E.g. (0, 1, 3, 5) will plot the zeroth, first, third and fifth light curves and classifications.
            If None or True, then all light curves will be plotted
        figdir : str
            Directory to save figure.
        light_curves : list
            This argument is only required if the get_predictions() method has not been run.
            Is a list of tuples. Each tuple contains the light curve information of a transient object in the form
            (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv).
            Here, mjd, flux, fluxerr, passband, and photflag are arrays.
            ra, dec, objid, redshift, and mwebv are floats

        """

        font = {'family': 'normal',
                'size': 33}
        matplotlib.rc('font', **font)

        if indexes_to_plot is None or indexes_to_plot is True:
            indexes_to_plot = np.arange(len(self.y_predict))

        if not hasattr(self, 'y_predict'):
            self.get_predictions()

        for idx in indexes_to_plot:
            lc_data = self.orig_lc[idx]
            used_passbands = [pb for pb in self.passbands if pb in lc_data['passband']]

            new_t = np.concatenate([lc_data[lc_data['passband'] == pb]['time'].data for pb in used_passbands])
            new_t = np.sort(new_t[~np.isnan(new_t)])
            new_y_predict = []
            all_flux = list(lc_data[lc_data['passband'] == 'g']['flux']) + list(lc_data[lc_data['passband'] == 'r']['flux'])

            argmax = self.timesX[idx].argmax() + 1
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15),
                                           num="animation_classification_vs_time_{}".format(idx), sharex=True)

            ax1.legend(frameon=False, fontsize=33)
            ax2.legend(frameon=False, fontsize=23.5)  # , ncol=2)
            ax2.set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
            ax1.set_ylabel("Relative Flux")  # , fontsize=15)
            ax2.set_ylabel("Class Probability")  # , fontsize=18)
            ax1.set_ylim(min(all_flux), max(all_flux))
            ax2.set_ylim(0, 1)
            ax1.set_xlim(min(new_t), max(new_t))
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added
            plt.tight_layout()
            fig.subplots_adjust(hspace=0)
            ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
            ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)

            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=2, bitrate=1800)

            for classnum, classname in enumerate(self.class_names):
                new_y_predict.append(np.interp(new_t, self.timesX[idx][:argmax], self.y_predict[idx][:, classnum][:argmax]))

            def animate(i):
                for pbidx, pb in enumerate(used_passbands):
                    # ax1.plot(self.timesX[idx][:argmax][:int(i + 1)], self.X[idx][:, pbidx][:argmax][:int(i + 1)],
                    #          label=pb, c=PB_COLOR[pb], lw=3)  # , markersize=10, marker='.')

                    # dea = [self.orig_lc[idx][pb]['time'] < self.timesX[idx][:argmax][int(i)]]
                    if i + 1 >= len(new_t):
                        break

                    pbmask = lc_data['passband'] == pb
                    dea = [lc_data[pbmask]['time'] < new_t[int(i+1)]]

                    ax1.errorbar(np.array(lc_data[pbmask]['time'])[dea], np.array(lc_data[pbmask]['flux'])[dea],
                                 yerr=np.array(lc_data[pbmask]['fluxErr'])[dea], fmt='o', label=pb,
                                 c=PB_COLOR[pb], lw=3, markersize=10)

                for classnum, classname in enumerate(self.class_names):
                    # ax2.plot(self.timesX[idx][:argmax][:int(i + 1)],
                    #          self.y_predict[idx][:, classnum][:argmax][:int(i + 1)],
                    #          '-', label=classname, color=CLASS_COLOR[classname], linewidth=3)
                    ax2.step(new_t[:int(i + 1)],
                             new_y_predict[classnum][:int(i + 1)],
                             '-', label=classname, color=CLASS_COLOR[classname], linewidth=3, where='post')

                # Don't repeat legend items
                ax1.legend(frameon=False, fontsize=33)
                ax2.legend(frameon=False, fontsize=23.5)
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                by_label1 = OrderedDict(zip(labels1, handles1))
                by_label2 = OrderedDict(zip(labels2, handles2))
                ax1.legend(by_label1.values(), by_label1.keys(), frameon=False, fontsize=33, loc='lower right')
                ax2.legend(by_label2.values(), by_label2.keys(), frameon=False, fontsize=21.5, loc='center right')

            ani = animation.FuncAnimation(fig, animate, frames=len(new_t), repeat=True)
            savename = os.path.join(figdir, 'classification_vs_time_{}_step.mp4'.format(self.objids[idx]))
            ani.save(savename, writer=writer)

