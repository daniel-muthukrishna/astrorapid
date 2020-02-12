import os
import pickle

from astrorapid.read_snana_fits import read_light_curves_from_snana_fits_files


def get_data(get_data_func, class_num, data_dir, save_dir, passbands, known_redshift=True, nprocesses=1, redo=False):
    """
    Get data using some function.

    Parameters
    ----------
    get_data_func : func
        Function that returns light_curves and takes at least the arguments class_num, data_dir, save_dir, passbands.
        E.g. get_data_func = get_data_custom(class_num, data_dir, save_dir, passbands, known_redshift)
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
        Whether to correct the light curves for cosmological time dilation or not.
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

    return get_data_func(class_num, data_dir, save_dir, passbands, known_redshift, nprocesses, redo)


def get_data_from_snana_fits(class_num, data_dir='data/ZTF_20190512/', save_dir='data/saved_light_curves/',
                             passbands=('g', 'r'), known_redshift=True, nprocesses=1, redo=False):
    """
    Get data from SNANA fits data files.

    """
    save_lc_filepath = os.path.join(save_dir, f"lc_classnum_{class_num}.pickle")

    if os.path.exists(save_lc_filepath) and not redo:
        with open(save_lc_filepath, "rb") as fp:  # Unpickling
            light_curves = pickle.load(fp)
    else:
        class_dir = os.path.join(data_dir, 'ZTF_MSIP_MODEL{:02d}'.format(class_num))
        files = os.listdir(class_dir)

        head_files = []
        phot_files = []
        for file in files:
            filepath = os.path.join(data_dir, class_dir, file)
            if filepath.endswith('HEAD.FITS'):
                head_files.append(filepath)
                phot_files.append(filepath.replace('_HEAD.FITS', '_PHOT.FITS'))
            print(filepath)

        light_curves = read_light_curves_from_snana_fits_files(head_files, phot_files, passbands,
                                                               known_redshift=known_redshift, nprocesses=nprocesses)

        with open(save_lc_filepath, "wb") as fp:  # Pickling
            pickle.dump(light_curves, fp)

    return light_curves


