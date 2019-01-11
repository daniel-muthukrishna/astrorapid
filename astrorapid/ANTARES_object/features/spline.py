# -*- coding: UTF-8 -*-
"""
Functions for spline interpolation with LAobjectss

"""

from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
import scipy.interpolate as scinterp

class SplineMixin(object):

    def spline_smooth(self, per=None, minobs=10, phase_offset=None, recompute=False, smoothed=False, gpr=True):
        """
        per = fit a periodic spline
        minobs = mininum number of observations in each filter before we fit
        """
        outtck = getattr(self, 'outtck', None)

        if outtck is not None:
            if not recompute:
                return outtck
        else:
            outtck = {}

        filters = self.filters
        outlc = self.get_lc(recompute=recompute, per=per, phase_offset=phase_offset, smoothed=smoothed, gpr=gpr)

        # stick to cubic interpolation
        k = 3

        for i, pb in enumerate(filters):
            thislc = outlc.get(pb)
            thisphase, thisFlux, thisFluxErr = thislc
            nobs = len(thisphase)
            if nobs < minobs:
                continue
            m2 = thisphase.argsort()
            minphase = thisphase[m2].min()
            maxphase = thisphase[m2].max()

            # don't try and fit pathalogical lightcurves which all have
            # different mags measured at the same time
            if minphase==maxphase:
                continue

            # degree can be cubic but will get clipped if needed
            degree = k

            x = thisphase[m2]
            y = thisFlux[m2]

            x, m2 = np.unique(x, return_index=True)
            y = y[m2]

            cv = np.array(list(zip(x,y)))
            count = len(cv)

            # If periodic, extend the point array by count+degree+1
            if per > 0:
                factor, fraction = divmod(count+degree+1, count)
                cv = np.concatenate((cv,) * factor + (cv[:fraction],))
                count = len(cv)
                degree = np.clip(degree,1,degree)
            # If opened, prevent degree from exceeding count-1
            else:
                degree = np.clip(degree,1,count-1)

            # Calculate knot vector
            kv = None
            if per > 0:
                kv = np.arange(0-degree,count+degree+degree-1,dtype='int')
            else:
                kv = np.array([0]*degree + list(range(count-degree+1)) + [count-degree]*degree,dtype='int')

            # Calculate query range
            periodic = int(bool(per))
            u = np.linspace(periodic,(count-degree),count)
            tck = (kv, cv, degree)

            outtck[pb] = tck, u
        self.outtck = outtck
        return outtck

