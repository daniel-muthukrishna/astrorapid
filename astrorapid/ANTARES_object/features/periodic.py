# -*- coding: UTF-8 -*-
"""
Functions for the derivation of periodic features with LAobjects
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import warnings
from gatspy import periodic


class PeriodicMixin(object):
    """
    Features computed for periodic events
    """

    def periodscan(self, min_p, max_p, periods=None):
        """
        Computes a LombScargle periodgram from minp to maxp, or alternatively
        on the array of periods sets model, periods, P_multi and best_period,
        overwriting if they already exist
        """
        nobs = len(self.time)
        if nobs == 0:
            return None, None, None

        # if self.time.max() - self.time.min() < max_p:
        #     return None, None, None

        optimizer_kwds = {'quiet':True}
        try:
            if nobs <= 100:
                model = periodic.LombScargleMultiband(fit_period=True, optimizer_kwds=optimizer_kwds)
            else:
                print('nobs', nobs, self.objectId)
                model = periodic.LombScargleMultibandFast(fit_period=True, optimizer_kwds=optimizer_kwds)
            model.optimizer.period_range = (min_p, max_p)
            model.fit(self.time, self.flux, self.fluxErr, self.passband)

            if periods is None:
                periods, P_multi = model.periodogram_auto()
            else:
                P_multi = model.periodogram(periods)

            self.model = model
            self.best_period = model.best_period
            self.periods = periods
            self.P_multi = P_multi
            return model, periods, P_multi

        except Exception as e:
            message = '{}\nCould not run periodscan for {} at locus ID {}'.format(e, self.objectId, self.locusId)
            warnings.warn(message, RuntimeWarning)
            return None, None, None


    def phase_for_period(self, period, phase_offset=None):
        """
        Returns the phase for some period. Exposed directly to trial different
        periods, but ideally should be used by get_phase.
        """
        if phase_offset is None:
            phase_offset = 0.
        phase = ((self.time - phase_offset )/ period) % 1
        return phase


    def get_phase(self, per=None, phase_offset=None):
        """
        Returns the phase of the object
        if periodic, and best_period is set, simply returns phase with it
        if periodic, nad best_period is not set, computes a periodogram with some defaults, and returns it
        if not periodic, just returns phase relative to first event

        TODO - return phase relative to some time argument, and accept
        arguments for periodscan (maybe with *args, **kwargs)
        """
        phase = None
        if per is None:
            per = self.per # default is False
        if per:
            if self.best_period is None:
                model = getattr(self, 'model', None)
                if model is None:
                    # we don't have a model - calculate it
                    max_p = 100.
                    if self.time.max() - self.time.min() < 100:
                        max_p = self.time.max() - self.time.min()
                    model, periods, P_multi = self.periodscan(0.1, max_p)
                    if model is None:
                        # we failed to get a model - give up
                        return self.get_phase(per=None, phase_offset=phase_offset)
                    # we got a model - set the period
                    self.best_period = model.best_period
                    best_period = self.best_period
                    self.model = model
                else:
                    # we have a model - just restore the period from it
                    best_period = model.best_period
            else:
                # we have a period - just restore it
                best_period = self.best_period
            phase = self.phase_for_period(best_period, phase_offset=phase_offset)
        else:
            if phase_offset is None:
                phase_offset = 0.
            phase = self.time - phase_offset
        return phase


    def get_best_periods(self):
        model, periods, P_multi = self.periodscan(0.1, 100.)
        periods, scores = model.find_best_periods(5, return_scores=True)

        return periods, scores


