# -*- coding: UTF-8 -*-
"""
Functions for the derivation of base features with LAobjects
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np


class PlasticcMixin(object):
    """
    Methods specific to the Phootometric LSST Astronomical Time-series
    Classification Challenge (PLASTiCC)
    """
    def finalize_plasticc(self):
        if not 'photflag' in self._extra_cols:
            message = 'Object {} does not appear to be a valid PLAsTiCC LC. Must have `photflag`'.format(self.objectId)
            raise RuntimeError(message)

        trigger = (self.photflag == 6144)
        if len(self.time[trigger]) != 1:
            message = 'Object {} does not have a unique trigger'.format(self.objectId)
            raise RuntimeError(message)

        time_trigger   = self.time[trigger][0]
        deltaT_trigger = np.abs(self.time - time_trigger)

        for pb in self.filters:
            # find observations in each passband that are non-detections or trigger with time to trigger < season length
            indf = np.where((self.passband == pb) & ((self.photflag == 0) | (self.photflag == 6144)) & (deltaT_trigger < 90))
            if len(deltaT_trigger[indf]) > 0:
                best_ref = deltaT_trigger[indf].argmin()
            else:
                indf = np.where((self.passband == pb) & (self.photflag == 4096) & (deltaT_trigger < 90))
                if len(deltaT_trigger[indf]) > 0:
                    best_ref = deltaT_trigger[indf].argmin()
                else:
                    continue
            pb_ref = self.flux[indf][best_ref]
            key='{}Ref'.format(pb)
            self.header[key] = pb_ref 
