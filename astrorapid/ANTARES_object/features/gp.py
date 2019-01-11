# -*- coding: UTF-8 -*-
"""
Functions for Gaussian Process Regression with  LAobjects

"""

from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
import george
from george import kernels
import scipy.optimize as op


class GPMixin(object):
    """
    Methods to smooth lightcurves with a Gaussian Process
    """
    def gaussian_process_smooth(self, per=None, minobs=10, phase_offset=None, recompute=False, scalemin=None, scalemax=None):
        """
        per = cheaty hackish thing to get a gaussian process with some
        continuity at the end points

        minobs = mininum number of observations in each filter before we fit
        """
        outgp = getattr(self, 'outgp', None)

        if outgp is not None:
            if not recompute:
                return outgp
        else:
            outgp = {}

        # note that we explicitly ask for non-smoothed and non-gpr lc here, since we're trying to compute the gpr
        outlc = self.get_lc(recompute=False, per=per, smoothed=False, gpr=False, phase_offset=phase_offset)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thisphase, thisFlux, thisFluxErr = thislc

            nobs = len(thisphase)

            # if we do not have enough observations in this passband, skip it
            if nobs < minobs:
                continue

            # TODO : REVISIT KERNEL CHOICE
            if per == 1:
                # periodic
                kernel = kernels.ConstantKernel(1.)*kernels.ExpSine2Kernel(1.0, 0.0)
            elif per == 2:
                # quasiperiodic
                kernel = kernels.ConstantKernel(1.)*kernels.ExpSquaredKernel(100.)*kernels.ExpSine2Kernel(1.0, 0.0)
            else:
                # non-periodic
                kernel = kernels.ConstantKernel(1.)*kernels.ExpSquaredKernel(1.)

            gp = george.GP(kernel, mean=thisFlux.mean(), fit_mean=True, fit_white_noise=True, white_noise=np.log(thisFluxErr.mean()**2.))

            # define the objective function
            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.lnlikelihood(thisFlux, quiet=True)
                return -ll if np.isfinite(ll) else 1e25

            # define the gradient of the objective function.
            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_lnlikelihood(thisFlux, quiet=True)

            # pre-compute the kernel
            gp.compute(thisphase, thisFluxErr)
            p0 = gp.get_parameter_vector()

            max_white_noise = np.log((3.*np.median(thisFluxErr))**2.)
            min_white_noise = np.log((0.3*np.median(thisFluxErr))**2)

            # coarse optimization with scipy.optimize
            # TODO : almost anything is better than scipy.optimize
            if per ==1:
                # mean, white_noise, amplitude, gamma, FIXED_period
                results = op.minimize(nll, p0, jac=grad_nll, bounds=[(None, None), (min_white_noise, max_white_noise),\
                        (None, None), (None, None), (0.,0.)])
            elif per==2:
                # mean white_noise, amplitude, variation_timescale, gamma, FIXED_period
                results = op.minimize(nll, p0, jac=grad_nll, bounds=[(None, None), (min_white_noise, max_white_noise),\
                        (None, None), (scalemin, scalemax), (None, None), (0.,0.)])
            else:
                # mean white_noise, amplitude, variation_timescale
                results = op.minimize(nll, p0, jac=grad_nll, bounds=[(None, None), (min_white_noise, max_white_noise),\
                        (None, None), (scalemin,scalemax)])

            gp.set_parameter_vector(results.x)
            # george is a little different than sklearn in that the prediction stage needs the input data
            outgp[pb] = (gp, thisphase, thisFlux, thisFluxErr)

        self.outgp = outgp
        return outgp
