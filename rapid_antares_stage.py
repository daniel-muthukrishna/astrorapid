import numpy as np
from astrorapid.classify import Classify


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

    alert_id, mjd, ras, decs, passband, mag, magerr, zeropoint, = locus_data.get_time_series('ra', 'dec', 'ztf_fid',
                                                                                             'ztf_magpsf',
                                                                                             'ztf_sigmapsf',
                                                                                             'ztf_magzpsci')

    # Ignore mag Nonetype values
    alertid, mjd, ras, decs, passband, mag, magerr, zeropoint = delete_indexes(np.where(mag == None), alert_id, mjd,
                                                                               ras, decs, passband, mag, magerr,
                                                                               zeropoint)

    print(locus_data.get_time_series('ra', 'dec', 'ztf_fid', 'ztf_magpsf', 'ztf_sigmapsf', 'ztf_magzpsci'))
    if len(mjd) < 3:
        print("less than 3 points")
        return
    zpt_median = np.median(zeropoint[zeropoint != None])
    zeropoint[zeropoint == None] = zpt_median
    zeropoint = np.asarray(zeropoint, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)

    flux = 10. ** (-0.4 * (mag - zeropoint))
    fluxerr = np.abs(flux * magerr * (np.log(10.) / 2.5))

    passband = np.where((passband == 1) | (passband == '1.0'), 'g', passband)
    passband = np.where((passband == 2) | (passband == '2.0'), 'r', passband)

    # Set photflag detections when S/N > 5
    photflag = np.zeros(len(flux))
    photflag[flux / fluxerr > 5] = 4096
    photflag[np.where(mjd == min(mjd[photflag == 4096]))] = 6144

    deleteindexes = np.where((passband == 3) | (passband == '3.0') | (np.isnan(mag)))
    mjd, passband, flux, fluxerr, zeropoint, photflag = delete_indexes(deleteindexes, mjd, passband, flux, fluxerr,
                                                                       zeropoint, photflag)

    light_curve_list = [(mjd, flux, fluxerr, passband, zeropoint, photflag, ra, dec, objid, redshift, mwebv)]

    from keras import backend as K
    K.clear_session()

    classification = Classify(known_redshift=True)
    predictions = classification.get_predictions(light_curve_list)
    print(predictions)

    for i, name in enumerate(classification.class_names):
        locus_data.set_property('rapid_class_probability_{}'.format(name), predictions[0][-1][i])

#     classification.plot_light_curves_and_classifications()
#     classification.plot_classification_animation()



# alert_id, mjd, ras, decs, passband, mag = np.array([[50628, 76964, 83220, 101280, 131368, 210180, 343972, 409192, 640237, 755152,
#   893450, 1140978, 1462313, 1561104, 1566501, 1731769, 1907015, 2146976, 2184489
# , 2455729, 2612737, 2607451, 2760548, 2918382]
# , [58275.4583333, 58277.3702778001, 58277.3936573998, 58278.4133333
# , 58279.3699536999, 58282.4010068998, 58286.4203471998, 58288.4130902998
# , 58299.3970255, 58305.3770833001, 58316.3791898, 58321.4013657002
# , 58342.3261806001, 58345.3099884, 58345.3198147998, 58350.3624073998
# , 58357.2070138999, 58363.1932406998, 58363.2334027998, 58366.2684259
# , 58368.2530439999, 58368.2777661998, 58370.2472569002, 58372.2575463001],
# [303.5818722, 303.5818825, 303.5819365, 303.5819067, 303.5818955, 303.5817949,
# 303.5818342, 303.5818464, 303.5817169, 303.5818453, 303.581839, 303.5817122,
# 303.5820197, 303.5817131, 303.5819175, 303.5819107, 303.5819203, 303.581827,
# 303.5818347, 303.5819155, 303.5818597, 303.5819601, 303.5818241, 303.5818872],
# [23.5718292, 23.5718516, 23.5718345, 23.5718284, 23.5717546, 23.5717519
# , 23.5718695, 23.571736, 23.5717482, 23.5718123, 23.5717901, 23.5717176
# , 23.5718903, 23.5717558, 23.5718063, 23.571838, 23.571838, 23.5717979
# , 23.5717624, 23.5718559, 23.571858, 23.5718487, 23.5717666, 23.5717776],
# [1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1],
# [19.3178405761719, 18.8213596343994, 17.6933555603027, 19.2889881134033
# , 19.5614376068115, 19.3227100372314, 19.4276275634766, 18.6863403320312
# , 17.6444511413574, 17.6018676757812, 18.7111587524414, 19.5735454559326
# , 18.7006359100342, 18.6792221069336, 17.9845218658447, 17.6935539245605
# , 18.1521644592285, 18.5742492675781, 18.3022956848145, 19.352014541626
# , 18.664083480835, 18.2836437225342, 19.5756530761719, 19.2778835296631]])
