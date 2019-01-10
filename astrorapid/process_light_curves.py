import os
from collections import OrderedDict
import numpy as np
import h5py
import multiprocessing as mp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
import pandas as pd
import json
import argparse

from helpers import calc_luminosity
from LAobject import LAobject


def process_light_curve(mjd, flux, fluxerr, passband, zeropoint, photflag, ra, dec, objid, redshift=None, mwebv=None):
    """
    Process light curves. Each argument is a 1D array.
    The passband array indicates which passband each measurement corresponds to.
    """

    # Get galactic latitude. Later make cuts on this
    c_icrs = SkyCoord(ra=header['ra'] * u.degree, dec=header['dec'] * u.degree, frame='icrs')
    b = c_icrs.galactic.b.value

    # Get trigger alert mjd
    trigger_mjd = float(mjd[photflag == 1][0])
    t = mjd - trigger_mjd

    # Account for distance and time dilation if redshift is known
    if redshift is not None:
        t = t / (1 + redshift)
        dlmu = cosmo.distmod(redshift)
        flux, fluxerr = calc_luminosity(flux, fluxerr, dlmu)

    obsid = np.arange(len(t))

    laobject = LAobject(locusId=objid, objectId=objid, time=t, flux=flux, fluxErr=fluxerr, obsId=obsid,
                        passband=passband, zeropoint=zeropoint, per=False, mag=False, photflag=photflag,
                        z=redshift)
    outlc = laobject.get_lc(recompute=True)

    otherinfo = [redshift, mwebv, trigger_mjd]

    savepd = {
    pb: pd.DataFrame(lcinfo).loc[[0, 5, 6, 7]].rename({0: 'time', 5: 'fluxRenorm', 6: 'fluxErrRenorm', 7: 'photflag'}).T
    for pb, lcinfo in
    outlc.items()}  # Convert to dataframe rows: time, fluxNorm, fluxNormErr, photFlag; columns: ugrizY
    savepd['otherinfo'] = pd.DataFrame(otherinfo)
    savepd = pd.DataFrame(
        {(outerKey, innerKey): values for outerKey, innerDict in savepd.items() for innerKey, values in
         innerDict.items()})  # Use multilevel indexing