# -*- coding: UTF-8 -*-
"""
Functions for the derivation of base features with LAobjects
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
import astropy.table as at


class BaseMixin(object):
    """
    Methods to derive baseline features for LAobjects
    """

    def get_lc(self, recompute=False):
        """
        Most often, we want a lightcurve broken up passband by passband to
        compute features. This stores the input light curve in that format.
        Return a lightcurve suitable for computing features
        """
        outlc = getattr(self, '_outlc', None)
        if outlc is None or recompute:
            filters = self.filters
            outlc = {}

            # store the indices of each filter
            for i, pb in enumerate(self.filters):
                mask = np.where(self.passband == pb)[0]
                m2 = self.time[mask].argsort()
                ind = mask[m2]
                if len(ind) > 0:
                    outlc[pb] = ind
            self._outlc = outlc

        # for each of the default_cols, get the elements for t passband
        out = {}
        for pb, ind in outlc.items():
            out[pb] = [getattr(self, column)[ind] for column in self._default_cols]

        return out

    def get_lc_as_table(self):
        columns = ['passband', 'time', 'fluxUnred', 'fluxErrUnred', 'photflag']
        rename_columns = ['passband', 'time', 'flux', 'fluxErr', 'photflag']
        out = [getattr(self, col) for col in columns]

        out_table = at.Table(out, names=rename_columns)
        out_table = out_table.group_by('passband')

        return out_table


