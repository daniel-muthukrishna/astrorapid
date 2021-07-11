import numpy as np
from astrorapid.prepare_arrays import PrepareArrays


class PrepareInputArrays(PrepareArrays):
    def __init__(self, passbands=('g', 'r'), contextual_info=('redshift',), bcut=True, zcut=None,
                 nobs=50, mintime=-70, maxtime=80, timestep=3.0):
        PrepareArrays.__init__(self, passbands, contextual_info, nobs, mintime, maxtime, timestep)
        self.bcut = bcut
        self.zcut = zcut

    def prepare_input_arrays(self, lightcurves):
        nobjects = len(lightcurves)

        X = np.zeros(shape=(nobjects, self.nfeatures, self.nobs))
        timesX = np.zeros(shape=(nobjects, self.nobs))
        objids_list = []
        orig_lc = []
        deleterows = []
        trigger_mjds = []

        for i, (objid, data) in enumerate(lightcurves.items()):
            print("Preparing light curve {} of {}".format(i, nobjects))

            redshift = data.meta['redshift']
            b = data.meta['b']
            trigger_mjd = data.meta['trigger_mjd']

            # Make cuts
            deleterows, deleted = self.make_cuts(data, i, deleterows, b, redshift, class_num=None, bcut=self.bcut,
                                                 zcut=self.zcut, pre_trigger=False)
            if deleted:
                continue

            tinterp, len_t = self.get_t_interp(data)
            timesX[i][0:len_t] = tinterp
            orig_lc.append(data)
            objids_list.append(objid)
            trigger_mjds.append(trigger_mjd)
            X = self.update_X(X, i, data, tinterp, len_t, objid, self.contextual_info, data.meta)

        deleterows = np.array(deleterows)
        if len(deleterows) > 0:
            X = np.delete(X, deleterows, axis=0)
            timesX = np.delete(timesX, deleterows, axis=0)

        # Correct shape for keras is (N_objects, N_timesteps, N_passbands) (where N_timesteps is lookback time)
        X = X.swapaxes(2, 1)

        return X, orig_lc, timesX, objids_list, trigger_mjds