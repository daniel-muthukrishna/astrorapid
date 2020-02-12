from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import pandas as pd

from astrorapid import helpers, model_early_lightcurve
from astrorapid.ANTARES_object.LAobject import LAobject


class InputLightCurve(object):
    def __init__(self, mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift=None, mwebv=None,
                 known_redshift=True, training_set_parameters=None):
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
        known_redshift : bool
            Whether to use redshift in processing the light curves and making the arrays.
        training_set_parameters : dict
            Optional parameter. If this is not None, then determine the explosion time, t0, for full the training set.
            The dictionary must have the following keys: {class_number, peakmjd}
        """

        self.mjd = np.array(mjd)
        self.flux = np.array(flux)
        self.fluxerr = np.array(fluxerr)
        self.passband = np.array(passband)
        self.photflag = np.array(photflag)
        self.ra = ra
        self.dec = dec
        self.objid = objid
        self.redshift = redshift
        self.mwebv = mwebv
        self.known_redshift = known_redshift
        self.training_set_parameters = training_set_parameters
        if training_set_parameters is not None:
            self.class_number = training_set_parameters['class_number']
            self.peakmjd = training_set_parameters['peakmjd']

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

    def compute_t0(self, outlc):
        """ Calculate the explosion time for the trianing set if certain conditions are met. """
        calc_params = True
        inrange_mask = self.t < self.peakmjd
        if int(self.class_number) in [70, 80, 82, 83]:  # No t0 if model types (AGN, RRlyrae, Eclipsing Binaries)
            calc_params = False
        elif self.peakmjd - self.trigger_mjd < 0:  # No t0 if trigger to peak time is small
            calc_params = False
        elif len(self.t[inrange_mask]) < 3:  # No t0 if not At least 3 points before peak
            calc_params = False
        elif len(self.t[self.t < 0]) < 3:  # No t0 if there are less than three points before trigger
            calc_params = False

        if calc_params:
            earlytime = self.peakmjd - self.trigger_mjd
            fit_func, parameters = model_early_lightcurve.fit_early_lightcurve(outlc, earlytime)
        else:
            parameters = {pb: [-99, -99, -99] for pb in self.passband}

        t0 = parameters[next(iter(parameters))][2]  # Get t0 from any of the passbands

        return t0

    def preprocess_light_curve(self):
        """ Preprocess light curve. """

        # Account for distance and time dilation if redshift is known
        if self.known_redshift and self.redshift is not None:
            self.t = self.correct_time_dilation(self.t)
            # self.flux, self.fluxerr = self.correct_for_distance(self.flux, self.fluxerr)

        obsid = np.arange(len(self.t))

        laobject = LAobject(locusId=self.objid, objectId=self.objid, time=self.t, flux=self.flux, fluxErr=self.fluxerr,
                            obsId=obsid, passband=self.passband, per=False, mag=False,
                            photflag=self.photflag, z=self.redshift, mwebv=self.mwebv)

        outlc = laobject.get_lc_as_table()
        outlc.meta = {'redshift': self.redshift, 'b': self.b, 'mwebv': self.mwebv, 'trigger_mjd': self.trigger_mjd}

        if self.training_set_parameters is not None:
            t0 = self.compute_t0(outlc)
            outlc.meta['t0'] = t0
            outlc.meta['peakmjd'] = self.peakmjd

        return outlc


def read_multiple_light_curves(light_curve_list, known_redshift=True, training_set_parameters=None):
    """
    light_curve_list is a list of tuples with each tuple having entries:
    mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv

    Returns the processed light curves
    """

    processed_light_curves = {}
    for light_curve in light_curve_list:
        mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv = light_curve
        inputlightcurve = InputLightCurve(*light_curve, known_redshift=known_redshift,
                                          training_set_parameters=training_set_parameters)
        processed_light_curves[objid] = inputlightcurve.preprocess_light_curve()

    return processed_light_curves
