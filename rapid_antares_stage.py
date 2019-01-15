import numpy as np
from astrorapid.classify import Classify


def rapid_stage(locus_data):
    locus_properties = locus_data.get_properties()
    objid = locus_properties['alert_id']
    ra = locus_properties['ra']
    dec = locus_properties['dec']
    redshift = 0.
    mwebv = 0.2

    id, mjd, flux, fluxerr, passband, mag = locus_data.get_time_series('ra', 'dec', 'ztf_fid', 'ztf_magpsf')
    passband = np.where(passband == 1, 'g', passband)
    passband = np.where(passband == 2, 'r', passband)
    zeropoint = [27.5] * len(mjd)
    photflag = [6144] + [4096] * (len(mjd) - 1)

    light_curve_list = [(mjd, flux, fluxerr, passband, zeropoint, photflag, ra, dec, objid, redshift, mwebv)]

    classification = Classify(light_curve_list)
    predictions = classification.get_predictions()
    print(predictions)

    # classification.plot_light_curves_and_classifications()
    # classification.plot_classification_animation()
