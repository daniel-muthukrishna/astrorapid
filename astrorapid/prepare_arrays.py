import numpy as np
from scipy.interpolate import interp1d

from astrorapid import helpers

# fix random seed for reproducibility
np.random.seed(42)


class PrepareArrays(object):
    def __init__(self, passbands=('g', 'r'), contextual_info=('redshift',), nobs=50, mintime=-70, maxtime=80,
                 timestep=3.0):
        self.passbands = passbands
        self.contextual_info = contextual_info
        self.npassbands = len(passbands)
        self.nfeatures = self.npassbands + len(self.contextual_info)
        self.nobs = nobs
        self.timestep = timestep
        self.mintime = mintime
        self.maxtime = maxtime

    def make_cuts(self, data, i, deleterows, b, redshift=None, class_num=None, bcut=True, zcut=0.5, ignore_classes=(),
                  pre_trigger=True):
        deleted = False
        try:
            time = data[data['passband']=='r']['time'].data
        except KeyError:
            print("No r band data. passbands")
            deleterows.append(i)
            deleted = True
            return deleterows, deleted

        if len(data) < 4:
            print("Less than 4 epochs. nobs = {}".format(len(data)))
            deleterows.append(i)
            deleted = True
        elif pre_trigger and len(time[time < 0]) < 3:
            print("Less than 3 points in the r band pre trigger", len(time[time < 0]))
            deleterows.append(i)
            deleted = True
        elif bcut and abs(b) < 15:
            print("In galactic plane. b = {}".format(b))
            deleterows.append(i)
            deleted = True
        elif redshift is not None and zcut and (redshift > self.zcut or redshift == 0):
            print("Redshift cut. z = {}".format(redshift))
            deleterows.append(i)
            deleted = True
        elif class_num in ignore_classes:
            print("Not including class:", class_num)
            deleterows.append(i)
            deleted = True

        return deleterows, deleted

    def get_min_max_time(self, data):
        # Get min and max times for tinterp
        mintimes = []
        maxtimes = []
        for j, pb in enumerate(self.passbands):
            pbmask = data['passband']==pb
            time = data[pbmask]['time'].data
            try:
                mintimes.append(time.min())
                maxtimes.append(time.max())
            except ValueError:
                print("No data for passband: ", pb)
        mintime = min(mintimes)
        maxtime = max(maxtimes) + self.timestep

        return mintime, maxtime

    def get_t_interp(self, data):
        mintime, maxtime = self.get_min_max_time(data)

        tinterp = np.arange(mintime, maxtime, step=self.timestep)
        len_t = len(tinterp)
        if len_t > self.nobs:
            tinterp = tinterp[(tinterp >= self.mintime)]
            len_t = len(tinterp)
            if len_t > self.nobs:
                tinterp = tinterp[:-(len_t - self.nobs)]
                len_t = len(tinterp)
        return tinterp, len_t

    def update_X(self, X, i, data, tinterp, len_t, objid, contextual_info, meta_data):
        for j, pb in enumerate(self.passbands):
            # Drop infinite or nan values in any row
            data.remove_rows(np.where(~np.isfinite(data['time']))[0])
            data.remove_rows(np.where(~np.isfinite(data['flux']))[0])
            data.remove_rows(np.where(~np.isfinite(data['fluxErr']))[0])

            # Get data
            pbmask = data['passband']==pb
            time = data[pbmask]['time'].data
            flux = data[pbmask]['flux'].data
            fluxerr = data[pbmask]['fluxErr'].data
            photflag = data[pbmask]['photflag'].data

            # Mask out times outside of mintime and maxtime
            timemask = (time > self.mintime) & (time < self.maxtime)
            time = time[timemask]
            flux = flux[timemask]
            fluxerr = fluxerr[timemask]
            photflag = photflag[timemask]

            n = len(flux)  # Get vector length (could be less than nobs)

            if n > 1:
                # if flux[-1] > flux[-2]:  # If last values are increasing, then set fill_values to zero
                #     f = interp1d(time, flux, kind='linear', bounds_error=False, fill_value=0.)
                # else:
                #     f = interp1d(time, flux, kind='linear', bounds_error=False,
                #                  fill_value='extrapolate')  # extrapolate until all passbands finished.
                f = interp1d(time, flux, kind='linear', bounds_error=False, fill_value=0.)

                fluxinterp = f(tinterp)
                fluxinterp = np.nan_to_num(fluxinterp)
                fluxinterp = fluxinterp.clip(min=0)
                fluxerrinterp = np.zeros(len_t)

                for interp_idx, fluxinterp_val in enumerate(fluxinterp):
                    if fluxinterp_val == 0.:
                        fluxerrinterp[interp_idx] = 0
                    else:
                        nearest_idx = helpers.find_nearest(time, tinterp[interp_idx])
                        fluxerrinterp[interp_idx] = fluxerr[nearest_idx]

                X[i][j][0:len_t] = fluxinterp
                # X[i][j * 2 + 1][0:len_t] = fluxerrinterp

        # Add contextual information
        for jj, c_info in enumerate(contextual_info, 1):
            X[i][j + jj][0:len_t] = meta_data[c_info] * np.ones(len_t)

        return X
