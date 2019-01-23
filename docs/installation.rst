============
Installation
============

Using pip
---------
The easiest and preferred way to install DASH (to ensure the latest stable version) is using pip:

.. code-block:: bash

    pip install astrorapid --upgrade

From Source
-----------
Alternatively, the source code can be downloaded from GitHub by running the following command:

.. code-block:: bash

    git clone https://github.com/daniel-muthukrishna/astrorapid.git

Dependencies
------------
Using pip to install astrorapid will automatically install the mandatory dependencies: :code:`numpy`, :code:`tensorflow`, :code:`keras`, :code:`astropy`, :code:`pandas`, :code:`extinction`, and :code:`scikit-learn`.

If you wish to plot your classifications or train your own classifier with your data you will need matplotlib and scipy as well. These can be installed with

.. code-block:: bash

    pip install matplotlib
    pip install scipy


Platforms
---------
RAPID can be run on both Python 2 and Python 3 distributions.
It is also cross-platform, working on Mac, Windows, and Linux distributions.
If you have any issues, please submit an issue at https://github.com/daniel-muthukrishna/astrorapid/issues.
