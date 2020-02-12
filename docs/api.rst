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

* :func:`astrorapid.get_custom_data.get_custom_data` - Read custom data files and save preprocessed light curves.
* :func:`astrorapid.custom_classifier.create_custom_classifier` - Run preprocessing and train neural network classification model.


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


.. automodule:: astrorapid.get_custom_data
    :members:

------

astrorapid.train_neural_network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: astrorapid.custom_classifier
    :members:


------

astrorapid.classifier_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: astrorapid.classifier_metrics
    :members:









