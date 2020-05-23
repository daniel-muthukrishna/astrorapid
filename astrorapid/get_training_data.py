import os
import pickle
import numpy as np

from astrorapid.read_snana_fits import read_light_curves_from_snana_fits_files
from astrorapid.helpers import delete_indexes
from astrorapid.process_light_curves import InputLightCurve


def get_data(get_data_func, class_num, data_dir, save_dir, passbands, known_redshift=True, nprocesses=1, redo=False,
             calculate_t0=True):
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
    calculate_t0 : bool
        Whether to calculate t0 during preprocessing.

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

    return get_data_func(class_num, data_dir, save_dir, passbands, known_redshift, nprocesses, redo, calculate_t0)


def get_data_from_snana_fits(class_num, data_dir='data/ZTF_20190512/', save_dir='data/saved_light_curves/',
                             passbands=('g', 'r'), known_redshift=True, nprocesses=1, redo=False, calculate_t0=True):
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
            filepath = os.path.join(class_dir, file)
            if filepath.endswith('HEAD.FITS'):
                head_files.append(filepath)
                phot_files.append(filepath.replace('_HEAD.FITS', '_PHOT.FITS'))
            print(filepath)

        light_curves = read_light_curves_from_snana_fits_files(head_files, phot_files, passbands,
                                                               known_redshift=known_redshift, nprocesses=nprocesses,
                                                               calculate_t0=calculate_t0)

        with open(save_lc_filepath, "wb") as fp:  # Pickling
            pickle.dump(light_curves, fp)

    return light_curves


def get_real_ztf_training_data(class_name, data_dir='data/real_ZTF_data_from_osc',
                               save_dir='data/saved_light_curves/', pbs=('g', 'r'),
                               known_redshift=True, nprocesses=1, redo=False, calculate_t0=True):
    """
    Get data from saved real ZTF data with names and types from the Open Supernova Catalog
    """

    save_lc_filepath = os.path.join(save_dir, f"lc_classnum_{class_name}.pickle")
    if os.path.exists(save_lc_filepath) and not redo:
        with open(save_lc_filepath, "rb") as fp:  # Unpickling
            light_curves = pickle.load(fp)
    else:
        light_curves = {}
        data_filepath = os.path.join(data_dir, f"ZTF_data_{class_name}_osc-6-May-2020.pickle")
        with open(data_filepath, "rb") as fp:
            mjds, passbands, mags, magerrs, photflags, zeropoints, dc_mags, dc_magerrs, magnrs, \
            sigmagnrs, isdiffposs, ras, decs, objids, redshifts, mwebvs = pickle.load(fp)

        for i, objid in enumerate(objids):
            if known_redshift and (redshifts[i] is None or np.isnan(redshifts[i])):
                print(f"Skipping {objid} because redshift is unknown and known_redshift model is selected")
                continue

            flux = 10. ** (-0.4 * (mags[i] - zeropoints[i]))
            fluxerr = np.abs(flux * magerrs[i] * (np.log(10.) / 2.5))

            passbands[i] = np.where((passbands[i] == 1) | (passbands[i] == '1'), 'g', passbands[i])
            passbands[i] = np.where((passbands[i] == 2) | (passbands[i] == '2'), 'r', passbands[i])

            mjd_first_detection = min(mjds[i][photflags[i] == 4096])
            photflags[i][np.where(mjds[i] == mjd_first_detection)] = 6144

            deleteindexes = np.where(((passbands[i] == 3) | (passbands[i] == '3')) | ((mjds[i] > mjd_first_detection) & (photflags[i] == 0)) | (np.isnan(flux)))
            if deleteindexes[0].size > 0:
                print("Deleting indexes {} at mjd {} and passband {}".format(deleteindexes, mjds[i][deleteindexes], passbands[i][deleteindexes]))
            mjd, passband, flux, fluxerr, zeropoint, photflag = delete_indexes(deleteindexes, mjds[i], passbands[i], flux, fluxerr, zeropoints[i], photflags[i])
            peakmjd = mjd[np.argmax(flux)]

            inputlightcurve = InputLightCurve(mjd, flux, fluxerr, passband, photflag,
                                              ras[i], decs[i], objid, redshifts[i], mwebvs[i],
                                              known_redshift=known_redshift,
                                              training_set_parameters={'class_number': class_name,
                                                                       'peakmjd': peakmjd},
                                              calculate_t0=calculate_t0)
            light_curves[objid] = inputlightcurve.preprocess_light_curve()

        with open(save_lc_filepath, "wb") as fp:
            pickle.dump(light_curves, fp)

    return light_curves
