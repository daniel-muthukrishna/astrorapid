=====
Usage
=====

Classify Light curves
+++++++++++++++++++++

Use the following example code:

.. code-block:: python

    from astrorapid.classify import Classify

    # Each light curve should be a tuple in this form. Look at the example code for an example of the input format.
    light_curve_info1 = (mjd, flux, fluxerr, passband, zeropoint, photflag, ra, dec, objid, redshift, mwebv)
    light_curve_list = [light_curve_info1,]

    # Classify Light curves
    classification = Classify(light_curve_list, known_redshift=True)
    predictions = classification.get_predictions()
    print(predictions)

    # Plot light curve and classification vs time of the light curves at the specified indexes
    classification.plot_light_curves_and_classifications(indexes_to_plot=(0,1,4,6))
    classification.plot_classification_animation(indexes_to_plot=(0,1,4,6))


Train your own classifier with your own data
++++++++++++++++++++++++++++++++++++++++++++
This can be achieve by running :code:`train_neural_network.py`.
More information on this will be added soon... Contact the author for support.
