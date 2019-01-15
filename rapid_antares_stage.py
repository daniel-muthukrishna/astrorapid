import numpy as np
from astrorapid.classify import Classify


def rapid_stage(locus_data):
    locus_properties = locus_data.get_properties()
    objid = locus_properties['alert_id']
    ra = locus_properties['ra']
    dec = locus_properties['dec']
    redshift = 0.
    mwebv = 0.2

    alert_id, mjd, ras, decs, passband, mag, magerr, zeropoint, = locus_data.get_time_series('ra', 'dec', 'ztf_fid', 'ztf_magpsf', 'ztf_sigmapsf', 'ztf_magzpsci')

    flux = 10. ** (-0.4 * (mag - zeropoint))
    fluxerr = np.abs(flux * magerr * (np.log(10.) / 2.5))
    passband = np.where(passband == 1, 'g', passband)
    passband = np.where(passband == 2, 'r', passband)
    photflag = [0] * int(len(mjd) / 2 - 3) + [6144] + [4096] * int(len(mjd) / 2 + 2)

    print(locus_data.get_time_series('ra', 'dec', 'ztf_fid', 'ztf_magpsf', 'ztf_sigmapsf', 'ztf_magzpsci', 'ztf_diffmaglim'))

    light_curve_list = [(mjd, flux, fluxerr, passband, zeropoint, photflag, ra, dec, objid, redshift, mwebv)]

    classification = Classify(light_curve_list)
    predictions = classification.get_predictions()
    print(predictions)

    # classification.plot_light_curves_and_classifications()
    # classification.plot_classification_animation()
