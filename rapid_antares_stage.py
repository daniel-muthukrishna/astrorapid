import traceback
import time
import numpy as np
from astrorapid.classify import Classify


# Reset Tensorflow
from tensorflow.keras import backend as K
K.clear_session()


# Load model from disk
classification = Classify(bcut=True, zcut=None)


def delete_indexes(deleteindexes, *args):
    newarrs = []
    for arr in args:
        newarr = np.delete(arr, deleteindexes)
        newarrs.append(newarr)

    return newarrs


def rapid_stage(locus_data):
    locus_properties = locus_data.get_properties()
    objid = locus_properties['alert_id']
    ra = locus_properties['ra']
    dec = locus_properties['dec']
    redshift = 0.  # TODO: Get correct redshift
    mwebv = 0.2  # TODO: Get correct extinction

    # Get redshift from SDSS_gals
    known_redshift = False
    catalog_matches = locus_data.get_astro_object_matches()
    if 'sdss_gals' in catalog_matches:
        redshift = catalog_matches['sdss_gals'][0]['z']
        known_redshift = True

    # Get lightcurve data
    alert_id, mjd, passband, mag, magerr, zeropoint = \
        locus_data.get_time_series(
            'ztf_fid', 'ztf_magpsf', 'ztf_sigmapsf', 'ztf_magzpsci',
            require=['ztf_fid', 'ztf_magpsf', 'ztf_sigmapsf'],
        )

    # Require 2 unique passbands
    if len(np.unique(passband)) < 2:
        print("less than 2 bands")
        return

    # Ignore lightcurves shorter than 3
    if len(mjd) < 3:
        print("less than 3 points")
        return

    # Fill in missing zeropoint values
    zeropoint = np.asarray(zeropoint, dtype=float)
    zpt_median = np.median(zeropoint[(zeropoint != None) & (~np.isnan(zeropoint))])
    zeropoint[zeropoint == None] = zpt_median
    zeropoint[np.isnan(zeropoint)] = zpt_median
    zeropoint = np.asarray(zeropoint, dtype=np.float64)
    if np.any(np.isnan(zeropoint)):
        locus_data.report_error(
            tag='astrorapid_zeropoint_contains_nan',
            data={
                'alert_id': objid,
            },
        )
        return

    # Compute flux
    mag = np.asarray(mag, dtype=np.float64)
    flux = 10. ** (-0.4 * (mag - zeropoint))
    fluxerr = np.abs(flux * magerr * (np.log(10.) / 2.5))

    # Set photflag detections when S/N > 5
    photflag = np.zeros(len(flux))
    photflag[flux / fluxerr > 5] = 4096
    photflag[np.where(mjd == min(mjd[photflag == 4096]))] = 6144

    # Filter out unwanted bands and convert ztf_fid to strings 'g', 'r'
    passband = np.where((passband == 1) | (passband == '1.0'), 'g', passband)
    passband = np.where((passband == 2) | (passband == '2.0'), 'r', passband)
    mjd, passband, flux, fluxerr, zeropoint, photflag = delete_indexes(
        np.where((passband == 3) | (passband == '3.0') | (np.isnan(mag))),
        mjd, passband, flux, fluxerr, zeropoint, photflag
    )

    # Do classification
    light_curves = [
        (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv)
    ]
    try:
        classification.known_redshift = known_redshift
        predictions, time_steps = classification.get_predictions(light_curves, return_predictions_at_obstime=True)
    except ValueError:
        locus_data.report_error(
            tag='astrorapid_get_predictions_valueerror',
            data={
                'alert_id': objid,
                'traceback': traceback.format_exc(),
            },
        )
        return

    # Output
    if predictions:
        for i, name in enumerate(classification.class_names):
            # Store properties
            p = predictions[0][-1][i]
            locus_data.set_property('rapid_class_probability_{}'.format(name), p)

            # Send to output streams
            if name == 'Pre-explosion':
                continue
            if p > 0.6:
                stream = 'astrorapid_{}'.format(name.lower().replace('-', '_'))
                locus_data.send_to_stream(stream)
