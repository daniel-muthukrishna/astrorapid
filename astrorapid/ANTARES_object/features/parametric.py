# -*- coding: UTF-8 -*-
"""
Functions for parametric representations of LAobjects

"""

from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
from .. import constants
from iminuit import Minuit


class ParametricMixin(object):

    def bazin_fit(self, minpbobs=6, recompute=False):
        """
        Fit a Bazin model to the LAobject
        """
        bazin_params = getattr(self, 'bazin_params', None)
        if bazin_params is not None:
            if not recompute:
                return bazin_params

        bazin_params = {}
        trigger_time = self.trigger_time

        filters = self.filters
        outlc = self.get_lc(recompute=recompute)

        for i, pb in enumerate(filters):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            # enough observations in this band to attempt to fit a model
            # there's 5 parameters, so there should be at least 6
            nobs = len(ttime)
            if nobs < minpbobs:
                print(nobs, "Not enough obs")
                continue

            # demand at least 3 points before or at trigger
            pre_trigger = ttime <= trigger_time 
            npre_trigger = len(ttime[pre_trigger])
            if npre_trigger <= 3:
                print(npre_trigger, "Not enough obs before trigger")
                continue

            # demand at least 3 points after trigger
            npost_trigger = nobs - npre_trigger 
            if npost_trigger <= 3:
                print(npost_trigger, "Not enough obs after trigger")
                continue

            # make sure the points are actual detections, not saturated or non-detections 
            nd_ind  = (tphotflag == constants.NONDETECT_PHOTFLAG)
            sat_ind = (tphotflag == constants.BAD_PHOTFLAG)
            good_ind = ~(nd_ind | sat_ind)
            ngood = len(ttime[good_ind])
            if ngood <= 3:
                print(ngood, "not enough good observations")
                continue 

            # use only unsaturated measurements
            use_ind = ~sat_ind 
            x  = ttime[use_ind] - trigger_time
            y  = tFluxUnred[use_ind]
            dy = tFluxErrUnred[use_ind]

            # hardcode some limits on parameters
            limit_A = (1E-6,  np.percentile(y, 99))
            limit_B = (-2000., 2000.)
            limit_t0 = (-10., 1.)
            limit_trise = (1., 50.)
            limit_tfall = (1., 50.)

            # define functions to get flux and return likelihood
            def _bazin_model(A, B, t0, trise, tfall):
                """
                Simple parametric model from Bazin et al '09
                https://www.aanda.org/articles/aa/pdf/2009/21/aa11847-09.pdf
                Eqn. 1
                """
                f = A*(np.exp(-(x-t0)/tfall)/(1.+np.exp((x-t0)/trise))) + B
                return f

            def _bazin_likelihood(A, B, t0, trise, tfall):
                f = _bazin_model(A, B, t0, trise, tfall)
                chisq = np.sum(((y - f)**2.)/dy**2.)
                return chisq

            # construct Minuit instance
            m = Minuit(_bazin_likelihood,\
                    A= np.percentile(y, 75), B=0., t0=0., trise=15., tfall=20.,\
                    limit_A=limit_A, limit_B=limit_B, limit_t0=limit_t0, limit_trise=limit_trise, limit_tfall=limit_tfall,\
                    print_level=1)

            m.migrad()

            vals = m.values 
            tparams = []
            for param in m.parameters:
                tparams += [m.values[param], m.errors[param],]
            tparams += [m.fval,]
            bazin_params[pb] = tparams 
        return bazin_params



                





