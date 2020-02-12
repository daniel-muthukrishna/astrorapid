=====
Usage
=====

Classify Light curves
+++++++++++++++++++++

Use the following example code:

.. code-block:: python

    from astrorapid.classify import Classify

    # Each light curve should be a tuple in this form. Look at the example code for an example of the input format.
    light_curve_info1 = (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv)
    light_curve_list = [light_curve_info1,]

    # Classify Light curves
    classification = Classify(known_redshift=True)
    predictions = classification.get_predictions(light_curve_list)
    print(predictions)

    # Plot light curve and classification vs time of the light curves at the specified indexes
    classification.plot_light_curves_and_classifications(indexes_to_plot=(0,1,4,6))
    classification.plot_classification_animation(indexes_to_plot=(0,1,4,6))


Train your own classifier with your own data
++++++++++++++++++++++++++++++++++++++++++++
You'll simply need to run the function `create_custom_classifier` to get started with training your own classifier.
An example is shown below.

* :func:`astrorapid.custom_classifier.create_custom_classifier`

.. code-block:: python

    from astrorapid.train_your_own_network import create_custom_classifier

    script_dir = os.path.dirname(os.path.abspath(__file__))

    create_custom_classifier(get_data_func=astrorapid.get_training_data.get_data_from_snana_fits,
                             data_dir='data/ZTF_sims',
                             class_nums=(1, 2, 12, 14, 3, 13, 41, 43, 51, 60, 61, 62, 63, 64, 70),
                             reread_data=False,
                             contextual_info=('redshift',),
                             passbands=('g', 'r'),
                             retrain_network=False,
                             train_epochs=50,
                             zcut=0.5,
                             bcut=True,
                             variablescut=True,
                             nprocesses=None,
                             nchunks=10000,
                             aggregate_classes=False,
                             otherchange='',
                             training_set_dir='data/training_set_files',
                             save_dir='data/saved_light_curves',
                             fig_dir='data/training_set_files/Figures'),
                             plot=True
                             )

You'll need to write your own function `get_data_func` to read your data and use the `astrorapid` preprocessing tools.
Use the skeleton function here :func:`astrorapid.get_custom_data.get_custom_data`, or as rewritten below.

.. code-block:: python

    def get_custom_data(class_num, data_dir, save_dir, passbands, known_redshift, nprocesses, redo):
        """
        Get data from custom data files.
        You will need to write this function with the following skeleton function:

        Parameters
        ----------
        class_num : int
            Class number. E.g. SNIa is 1. See helpers.py for lookup table.
            E.g. class_num = 1
        data_dir : str
            Directory where data is stored
            E.g. data_dir='data/ZTF_20190512/'
        save_dir : str
            Directory to save processed data
            E.g. save_dir='data/saved_light_curves/'
        passbands : tuple
            Passbands to use.
            E.g. passbands=('g', 'r')
        known_redshift : bool
            Whether to correct the light curves for cosmological time dilation using redshift.
        nprocesses : int or None
            Number of processes to use
        redo : bool
            Whether to redo reading the data and saving the processed data.


        Returns
        -------
        light_curves : dict of astropy.table.Table objects
            e.g light_curves['objid1'] =
                passband   time       flux     fluxErr   photflag
                  str1   float32    float32    float32    int32
                -------- -------- ----------- ---------- --------
                       g -46.8942  -48.926975  42.277767        0
                       g -43.9352  -105.35379   72.97575        0
                       g -35.9161  -46.264206    99.9172        0
                       g -28.9377  -28.978344  42.417065        0
                       g -25.9787  109.886566   46.03949        0
                       g -15.0399    -80.2485   80.38155        0
                       g -12.0218    93.51743  113.21529        0
                       g  -6.9585   248.88364 108.606865        0
                       g  -4.0411   341.41498  47.765404        0
                       g      0.0    501.7441   45.37485     6144
                     ...      ...         ...        ...      ...
                       r  40.9147   194.32494  57.836903     4096
                       r  59.9162    67.59185   45.66463     4096
                       r  62.8976    80.85155  44.356197     4096
                       r  65.8974   28.174305   44.75049     4096
                       r  71.8966  -18.790287 108.049774     4096
                       r  74.9297  -3.1707647  125.15057     4096
                       r  77.9341 -11.0205965 125.784676     4096
                       r  80.8576   129.65466   69.99305     4096
                       r  88.8922  -14.259436  52.917866     4096
                       r 103.8734   27.178356 115.537704     4096

        """

        # If the data has already been run and processed load it. Otherwise read it and save it
        save_lc_filepath = os.path.join(save_dir, f"lc_classnum_{class_num}.pickle")
        if os.path.exists(save_lc_filepath) and not redo:
            with open(save_lc_filepath, "rb") as fp:  # Unpickling
                light_curves = pickle.load(fp)
        else:
            light_curves = {}
            # Read in data from data_dir and get the mjd, flux, fluxerr, passband, photflag as 1D numpy arrays for
            # each light curve. Get the ra, dec, objid, redshift, mwebv, model_num, peak_mjd as floats or strings.
            # Set whether you'd like to train a model with a known redshift or not. Set known_redshift as a boolean.

            # Enter your own data-reading code here that gets the mjds, fluxes, fluxerrs, passbands, photflags,
            # ras, decs, objids, redshifts, mwebvs, model_nums, peak_mjds for all the light curves from the data_dir

            # Once you have the required data information for each light curve, pass it into InputLightCurve with
            # something like the following code:
            for i, objid in enumerate(objids):
                inputlightcurve = InputLightCurve(mjds[i], fluxes[i], fluxerrs[i], passbands[i], photflags[i],
                                                  ras[i], decs[i], objids[i], redshifts[i], mwebvs[i],
                                                  known_redshift=known_redshift,
                                                  training_set_parameters={'class_number': int(class_num),
                                                                           'peakmjd': peakmjds[i]})
                light_curves[objid] = inputlightcurve.preprocess_light_curve()

            # If you think that reading the data is too slow, you may want to replace the for loop above with
            # multiprocessing. See the example function in get_training_data.py if you need help doing this.

            # Next, we save it:
            with open(save_lc_filepath, "wb") as fp:  # Pickling
                pickle.dump(light_curves, fp)

        return light_curves
