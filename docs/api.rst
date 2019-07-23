===
API
===

RAPID has a few important classes and methods.

Classify light curves
---------------------

* :class:`astrorapid.classify.Classify` - Setup light curve classification
* :func:`astrorapid.classify.Classify.get_predictions` - Classify light curves!


Plot classifications
--------------------
* :func:`astrorapid.classify.Classify.plot_light_curves_and_classifications` - Plot light curves and classifications
* :func:`astrorapid.classify.Classify.plot_classification_animation` - Plot animation of light curves and classifications


Train your own classifier
-------------------------

* :func:`astrorapid.read_from_database.read_light_curves_from_snana_fits.main` - Store SNANA lightcurves into a HDF5 file
* :func:`astrorapid.train_neural_network.main` - Run preprocessing and train neural network classification model


The full documentation of useful classes and methods can be found below.

Full Documentation
------------------

.. autoclass:: astrorapid.classify.Classify
    :members:


------


.. autoclass:: astrorapid.process_light_curves.InputLightCurve
    :members:


------


.. autofunction:: astrorapid.process_light_curves.read_multiple_light_curves


------


.. autoclass:: astrorapid.prepare_arrays.PrepareArrays


------


.. autoclass:: astrorapid.prepare_arrays.PrepareInputArrays


------


.. autoclass:: astrorapid.prepare_arrays.PrepareTrainingSetArrays


------

astrorapid.read_from_database.read_light_curves_from_snana_fits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. automodule:: astrorapid.read_from_database.read_light_curves_from_snana_fits
    :members:

.. autoclass:: astrorapid.read_from_database.read_light_curves_from_snana_fits.GetData
    :members:


------

astrorapid.train_neural_network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: astrorapid.train_neural_network
    :members:


------

astrorapid.classifier_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: astrorapid.classifier_metrics
    :members:









