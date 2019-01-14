from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import helpers
from ANTARES_object.LAobject import LAobject


class InputLightCurve(object):
    def __init__(self, mjd, flux, fluxerr, passband, zeropoint, photflag, ra, dec, objid, redshift=None, mwebv=None):
        """

        Parameters
        ----------
        mjd : array
            List of floating point mjd times for entire light curve in all passbands.
        flux : array
            List of floating point fluxes for entire light curve in all passbands.
        fluxerr : array
            List of floating point flux errors times for entire light curve in all passbands.
        passband : array
            List of passbands as strings for each point in the array.
        zeropoint : array
            List of floating point zeropoints for entire light curve in all passbands.
        photflag : array
            List of integer flags indicating whether the observation was a detection or non-detection for each
            point in the light curve.
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        objid : str
            String to identify the input object.
        redshift : float
            Optional parameter. Cosmological redshift.
        mwebv : float
            Optional parameter. Milky Way E(B-V) extinction.
        """

        self.mjd = np.array(mjd)
        self.flux = np.array(flux)
        self.fluxerr = np.array(fluxerr)
        self.passband = np.array(passband)
        self.zeropoint = np.array(zeropoint)
        self.photflag = np.array(photflag)
        self.ra = ra
        self.dec = dec
        self.objid = objid
        self.redshift = redshift
        self.mwebv = mwebv

        self.b = self.get_galactic_latitude()
        self.trigger_mjd, self.t = self.get_trigger_time()


    def get_galactic_latitude(self):
        c_icrs = SkyCoord(ra=self.ra * u.degree, dec=self.dec * u.degree, frame='icrs')
        b = c_icrs.galactic.b.value

        return b

    def get_trigger_time(self):
        trigger_mjd = float(self.mjd[self.photflag == 6144][0])
        t = self.mjd - trigger_mjd

        return trigger_mjd, t

    def correct_time_dilation(self, t):
        t = t / (1 + self.redshift)

        return t

    def correct_for_distance(self, flux, fluxerr):
        dlmu = cosmo.distmod(self.redshift).value
        flux, fluxerr = helpers.calc_luminosity(flux, fluxerr, dlmu)

        return flux, fluxerr

    def preprocess_light_curve(self):
        """ Preprocess light curve. """

        # Account for distance and time dilation if redshift is known
        if self.redshift is not None:
            self.t = self.correct_time_dilation(self.t)
            self.flux, self.fluxerr = self.correct_for_distance(self.flux, self.fluxerr)

        obsid = np.arange(len(self.t))

        laobject = LAobject(locusId=self.objid, objectId=self.objid, time=self.t, flux=self.flux, fluxErr=self.fluxerr,
                            obsId=obsid, passband=self.passband, zeropoint=self.zeropoint, per=False, mag=False,
                            photflag=self.photflag, z=self.redshift)

        outlc = laobject.get_lc(recompute=True)

        otherinfo = [self.redshift, self.b, self.mwebv, self.trigger_mjd, self.objid]

        savepd = {
        pb: pd.DataFrame(lcinfo).loc[[0, 5, 6, 7]].rename({0: 'time', 5: 'fluxRenorm', 6: 'fluxErrRenorm', 7: 'photflag'}).T
        for pb, lcinfo in
        outlc.items()}  # Convert to dataframe rows: time, fluxNorm, fluxNormErr, photFlag; columns: ugrizY
        savepd['otherinfo'] = pd.DataFrame(otherinfo)
        savepd = pd.DataFrame(
            {(outerKey, innerKey): values for outerKey, innerDict in savepd.items() for innerKey, values in
             innerDict.items()})  # Use multilevel indexing

        return savepd


def read_multiple_light_curves(light_curve_list):
    """
    light_curve_list is a list of tuples with each tuple having entries:
    mjd, flux, fluxerr, passband, zeropoint, photflag, ra, dec, objid, redshift, mwebv

    Returns the processed light curves
    """

    processed_light_curves = []
    for light_curve in light_curve_list:
        inputlightcurve = InputLightCurve(*light_curve)
        processed_light_curves.append(inputlightcurve.preprocess_light_curve())

    return processed_light_curves


def prepare_input_arrays(lightcurves, passbands=('g', 'r'), contextual_info=(0,)):

    nobjects = len(lightcurves)
    nobs = 50
    npassbands = 2
    nfeatures = npassbands + 1

    X = np.zeros(shape=(nobjects, nfeatures, nobs))
    timesX = np.zeros(shape=(nobjects, nobs))
    objids_list = []
    orig_lc = []

    for i, data in enumerate(lightcurves):
        objid = i
        print("Preparing light curve {} of {}".format(i, nobjects))

        otherinfo = data['otherinfo'].values.flatten()
        redshift, b, mwebv, trigger_mjd, objid = otherinfo[0:5]

        # Make cuts
        if abs(b) < 15:
            print("In galactic plane. b = {}".format(b))
        if data.shape[0] < 4:
            print("Less than 4 epochs. nobs = {}".format(data.shape))
        time = data['r']['time'][0:nobs].dropna()
        if len(time[time < 0]) < 3:
            print("Less than 3 points in the r band pre trigger", len(time[time < 0]))

        # Get min and max times for tinterp
        timestep = 3.0
        mintimes = []
        maxtimes = []
        for j, pb in enumerate(passbands):
            if pb not in data:
                continue
            time = data[pb]['time'][0:nobs].dropna()
            mintimes.append(time.min())
            maxtimes.append(time.max())
        if mintimes == []:
            print("No data for objid:", i)
            continue
        mintime = min(mintimes)
        maxtime = max(maxtimes) + timestep
        tinterp = np.arange(mintime, maxtime, step=timestep)
        len_t = len(tinterp)
        if len_t >= nobs:
            tinterp = tinterp[(tinterp >= -70) & (tinterp <= 80)]
            len_t = len(tinterp)
            if len_t >= nobs:
                tinterp = tinterp[tinterp <= 80]
                len_t = len(tinterp)
                if len_t >= nobs:
                    tinterp = tinterp[len_t - nobs:]
                    len_t = len(tinterp)

        timesX[i][0:len_t] = tinterp

        endtime = 1000
        j = 0

        orig_lc.append(data)
        objids_list.append(objid)

        for j, pb in enumerate(passbands):
            if pb not in data:
                print("No", pb, "in objid:", objid)
                continue
            time = data[pb]['time'][0:nobs].dropna()
            try:
                flux = data[pb]['fluxRenorm'][0:nobs].dropna()
                fluxerr = data[pb]['fluxErrRenorm'][0:nobs].dropna()
            except KeyError:
                flux = data[pb][5][0:nobs].dropna()
                fluxerr = data[pb][6][0:nobs].dropna()
            photflag = data[pb]['photflag'][0:nobs].dropna()

            n = len(flux)  # Get vector length (could be less than nobs)

            if n > 1:
                if flux.values[-1] > flux.values[-2]:  # If last values are increasing, then set fill_values to zero
                    f = interp1d(time, flux, kind='linear', bounds_error=False, fill_value=0.)
                else:
                    f = interp1d(time, flux, kind='linear', bounds_error=False,
                                 fill_value='extrapolate')  # extrapolate until all passbands finished.

                fluxinterp = f(tinterp)
                fluxinterp = np.nan_to_num(fluxinterp)
                fluxinterp = fluxinterp.clip(min=0)
                X[i][j][0:len_t] = fluxinterp

        # Add contextual information
        for jj, c_idx in enumerate(contextual_info, 1):
            try:
                X[i][j + jj][0:len_t] = otherinfo[c_idx] * np.ones(len_t)
            except Exception as e:
                X[i][j + jj][0:len_t] = otherinfo[c_idx].values[0] * np.ones(len_t)

        # Correct shape for keras is (N_objects, N_timesteps, N_passbands) (where N_timesteps is lookback time)
        X = X.swapaxes(2, 1)

        return X, orig_lc, timesX, objids_list
