"""
Example usage:

.. code-block:: bash

    python -m astrorapid.read_from_database.read_light_curves_from_snana_fits --offset 0
    --offsetnext 3 --nprocesses 8 --savename 'testing' --combinefiles

"""

import os
import numpy as np
import h5py
import multiprocessing as mp
import pandas as pd
import argparse
import astropy.table as at
import astropy.io.fits as afits
from collections import OrderedDict

from astrorapid.process_light_curves import InputLightCurve


class GetData(object):
    def __init__(self):
        """ Get light curves. """

        self.phot_fields = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR', 'ZEROPT', 'PHOTFLAG']
        self.phot_fields_dtypes = {'FLT': np.str_, 'PHOTFLAG': np.int_}

    def get_light_curve_array(self, phot_file, ptrobs_min, ptrobs_max, standard_zpt=27.5):
        """ Get lightcurve from fits file as an array - avoid some Pandas overhead

        Parameters
        ----------
        objid : str
            The object ID. E.g. objid='DDF_04_NONIa-0004_87287'
        ptrobs_min : int
            Min index of object in _PHOT.FITS.
        ptrobs_max : int
            Max index of object in _PHOT.FITS.

        Return
        -------
        phot_out: pandas DataFrame
            A DataFrame containing the MJD, FLT, FLUXCAL, FLUXCALERR, ZEROPT seperated by each filter.
            E.g. Access the magnitude in the z filter with phot_out['z']['MAG'].
        """

        try:
            phot_HDU = afits.open(phot_file, memmap=True)
        except Exception as e:
            message = f'Could not open photometry file {phot_file}'
            raise RuntimeError(message)

        phot_data = phot_HDU[1].data[ptrobs_min - 1:ptrobs_max]

        phot_dict = OrderedDict()
        filters = list(set(phot_data['FLT']))  # e.g. ['i', 'r', 'Y', 'u', 'g', 'z']
        dtypes = dict(self.phot_fields_dtypes)
        for f in filters:
            fIndexes = np.where(phot_data['FLT'] == f)[0]
            phot_dict[f] = OrderedDict()
            for pfield in self.phot_fields:
                if pfield == 'ZEROPT':
                    phot_dict[f][pfield] = np.repeat(standard_zpt, len(fIndexes))
                elif pfield == 'FLT':
                    true_zpt = phot_data['ZEROPT'][fIndexes]
                    nobs = len(true_zpt)
                    phot_dict[f][pfield] = np.repeat(f.strip(), nobs)
                else:
                    phot_dict[f][pfield] = phot_data[pfield][fIndexes]

                if not pfield in dtypes:
                    dtypes[pfield] = np.float64

        phot_out = pd.DataFrame(phot_dict)
        phot_HDU.close()
        del phot_HDU[1].data
        return phot_out

    def convert_pandas_lc_to_recarray_lc(self, phot, passbands=('u', 'g', 'r', 'i', 'z', 'Y')):
        """
        ANTARES_object not Pandas format broken up by passband
        """
        pbs = passbands
        # name mapping for the defaults in phot_fields
        # any other column is going to become just lowercase it's current name
        name_map = {'FLUXCAL': 'flux', 'FLUXCALERR': 'dflux', 'ZEROPT': 'zpt',\
                        'FLT': 'pb', 'MJD': 'mjd', 'PHOTFLAG': 'photflag'}

        out = None
        out_names = None

        for this_pb in phot:
            # do we know what this passband is
            # this is just a sanity test in case we accidentally retrieve dummy entries with passband = -9
            if this_pb not in pbs:
                continue

            this_pb_lc = phot.get(this_pb)
            if this_pb_lc is None:
                continue

            if out is None:
                out = np.rec.fromarrays(list(this_pb_lc))
                if out_names is None:
                    out_names = list(this_pb_lc.axes[0])
                    out_names = [name_map.get(x, x.lower()) for x in out_names]
            else:
                temp = np.rec.fromarrays(list(this_pb_lc))
                out = np.concatenate((out, temp), axis=-1)

        out.dtype.names = out_names
        return out


def read_light_curves_from_snana_fits_files(save_fname, head_files, phot_files, passbands=('g', 'r'), known_redshift=True):
    """ Save lightcurves from SNANA HEAD AND PHOT FITS files

    Parameters
    ----------
    save_fname : str
        Filename to save hdf5 file.
    head_files : list
        List of SNANA header filepaths (.HEAD files).
    phot_files : list
        List of SNANA photometric filepaths (.PHOT files). Must be in the same order as the phot files.
    passbands : tuple
        passband filters.
    known_redshift : bool
        Whether to use redshift during training.

    """

    getter = GetData()

    store = pd.HDFStore(save_fname)

    for fileidx, headfilepath in enumerate(head_files):
        print(fileidx, headfilepath)
        # Check that phot file correponds to head file
        assert phot_files[fileidx].split('_')[-2] == head_files[fileidx].split('_')[-2]
        header_HDU = afits.open(head_files[fileidx])
        header_data = header_HDU[1].data

        for i, head in enumerate(header_data):
            model_num = head['SIM_TYPE_INDEX']
            snid = head['SNID']
            objid = 'field_{}_base_{}'.format(model_num, snid)
            ptrobs_min = head['PTROBS_MIN']
            ptrobs_max = head['PTROBS_MAX']
            peakmag_g = head['SIM_PEAKMAG_g']
            peakmag_r = head['SIM_PEAKMAG_r']
            redshift = head['SIM_REDSHIFT_HOST']
            dlmu = head['SIM_DLMU']
            peakmjd = head['PEAKMJD']
            mwebv = head['MWEBV']
            mwebv_err = head['MWEBV_ERR']
            ra = head['RA']
            if 'DEC' in header_data.names:
                dec = head['DEC']
            else:
                dec = head['DECL']
            photoz = head['HOSTGAL_PHOTOZ']
            photozerr = head['HOSTGAL_PHOTOZ_ERR']
            print(i, len(header_data))
            phot_data = getter.get_light_curve_array(phot_files[fileidx], ptrobs_min, ptrobs_max)

            lc = getter.convert_pandas_lc_to_recarray_lc(phot_data, passbands=passbands)

            inputlightcurve = InputLightCurve(lc['mjd'], lc['flux'], lc['dflux'], lc['pb'], lc['photflag'], ra,
                                              dec, objid, redshift, mwebv, known_redshift=known_redshift,
                                              training_set_parameters={'class_number': int(model_num), 'peakmjd': peakmjd})

            # TODO: work out why some light curves fail mcmc
            try:
                savepd = inputlightcurve.preprocess_light_curve()
            except Exception as e:
                print("Failed on object", objid, e)
                continue
            store.append(objid, savepd)

    store.close()
    print("saved %s" % save_fname)


def combine_hdf_files(save_dir, combined_savename, training_set_dir):
    fnames = os.listdir(save_dir)
    fname_out = os.path.join(training_set_dir, combined_savename)
    output_file = h5py.File(fname_out, 'w')

    for n, f in enumerate(fnames):
        print(n, f)
        try:
            f_hdf = h5py.File(os.path.join(save_dir, f), 'r')
            for objid in f_hdf.keys():
                objid = objid.encode('utf-8')
                h5py.h5o.copy(f_hdf.id, objid, output_file.id, objid)
            f_hdf.close()
        except OSError as e:
            print("Failed to open file", "f")
            print(e)
    output_file.close()


def create_all_hdf_files(args):
    head_files, phot_files, i, save_dir, passbands, known_redshift = args
    fname = os.path.join(save_dir, 'lc_{}.hdf5'.format(i))
    read_light_curves_from_snana_fits_files(fname, head_files, phot_files, passbands, known_redshift)


def main():
    """
    Save light curves to HDF5 files.
    """

    passbands = ('g', 'r')
    data_release = 'ZTF_20190512'
    field = 'MSIP'
    model = '%'
    known_redshift = True
    dir_name = '/Users/danmuth/PycharmProjects/astrorapid/sim_lc/test_SNIa'

    head_files = []
    phot_files = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith('HEAD.FITS'):
                head_files.append(filepath)
                phot_files.append(filepath.replace('_HEAD.FITS', '_PHOT.FITS'))
            print(filepath)
    # head_files = np.sort(head_files)
    # phot_files = np.sort(phot_files)
    print("Number of head files = {}".format(len(head_files)))

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', "--offset", type=int)
    parser.add_argument('-n', "--offsetnext", type=int)
    parser.add_argument('-m', "--nprocesses", type=int, help='Number of multiprocessing processes. Default is 1.')
    parser.add_argument("--savename", type=str)
    parser.add_argument("--combinefiles", help="Only set this if after this action, all files will have been created.",
                        action='store_true')
    args = parser.parse_args()
    if args.offset is not None:
        offset = args.offset
    else:
        offset = 0
    if args.offsetnext is not None:
        offset_next = args.offsetnext
    else:
        offset_next = 10
    if args.nprocesses is not None:
        nprocesses = args.nprocesses
    else:
        nprocesses = 1
    if args.savename is not None:
        savename = args.savename
    else:
        savename = ""
    print(offset, offset_next)

    training_set_dir = 'training_set_files'
    save_dir = os.path.join(training_set_dir, 'saved_lc_{}_{}_{}'.format(field, data_release, savename))
    if not os.path.exists(save_dir) and offset == 0:
        os.makedirs(save_dir)

    # Multiprocessing
    i_list = np.arange(offset, offset_next)
    print(i_list)
    args_list = []
    file_list = os.listdir(save_dir)
    for i in i_list:
        if 'lc_{}.hdf5'.format(i) not in file_list:
            head_files_part = [head_files[i]]
            phot_files_part = [phot_files[i]]
            print(os.path.join(save_dir, 'lc_{}.hdf5'.format(i)))
            args_list.append((head_files_part, phot_files_part, i, save_dir, passbands, known_redshift))

    if nprocesses == 1:
        for args in args_list:
            create_all_hdf_files(args)
    else:
        pool = mp.Pool(nprocesses)
        results = pool.map_async(create_all_hdf_files, args_list)
        pool.close()
        pool.join()

    if args.combinefiles:
        combine_hdf_files(save_dir, 'saved_lc_{}_{}_{}.hdf5'.format(field, data_release, savename), training_set_dir)


if __name__ == '__main__':
    main()


# keys: dtype=(numpy.record, [('SUBSURVEY', 'S40'), ('SNID', 'S16'), ('IAUC', 'S16'), ('FAKE', '>i2'), ('RA', '>f8'), ('DEC', '>f8'), ('PIXSIZE', '>f4'), ('NXPIX', '>i2'), ('NYPIX', '>i2'), ('CCDNUM', '>i2'), ('SNTYPE', '>i4'), ('NOBS', '>i4'), ('PTROBS_MIN', '>i4'), ('PTROBS_MAX', '>i4'), ('MWEBV', '>f4'), ('MWEBV_ERR', '>f4'), ('REDSHIFT_HELIO', '>f4'), ('REDSHIFT_HELIO_ERR', '>f4'), ('REDSHIFT_FINAL', '>f4'), ('REDSHIFT_FINAL_ERR', '>f4'), ('VPEC', '>f4'), ('VPEC_ERR', '>f4'), ('HOSTGAL_NMATCH', '>i2'), ('HOSTGAL_NMATCH2', '>i2'), ('HOSTGAL_OBJID', '>i8'), ('HOSTGAL_PHOTOZ', '>f4'), ('HOSTGAL_PHOTOZ_ERR', '>f4'), ('HOSTGAL_SPECZ', '>f4'), ('HOSTGAL_SPECZ_ERR', '>f4'), ('HOSTGAL_RA', '>f8'), ('HOSTGAL_DEC', '>f8'), ('HOSTGAL_SNSEP', '>f4'), ('HOSTGAL_DDLR', '>f4'), ('HOSTGAL_CONFUSION', '>f4'), ('HOSTGAL_LOGMASS', '>f4'), ('HOSTGAL_LOGMASS_ERR', '>f4'), ('HOSTGAL_sSFR', '>f4'), ('HOSTGAL_sSFR_ERR', '>f4'), ('HOSTGAL_MAG_g', '>f4'), ('HOSTGAL_MAG_r', '>f4'), ('HOSTGAL_MAGERR_g', '>f4'), ('HOSTGAL_MAGERR_r', '>f4'), ('HOSTGAL_SB_FLUXCAL_g', '>f4'), ('HOSTGAL_SB_FLUXCAL_r', '>f4'), ('PEAKMJD', '>f4'), ('SEARCH_TYPE', '>i4'), ('SIM_MODEL_NAME', 'S32'), ('SIM_MODEL_INDEX', '>i2'), ('SIM_TYPE_INDEX', '>i2'), ('SIM_TYPE_NAME', 'S8'), ('SIM_TEMPLATE_INDEX', '>i4'), ('SIM_LIBID', '>i4'), ('SIM_NGEN_LIBID', '>i4'), ('SIM_NOBS_UNDEFINED', '>i4'), ('SIM_SEARCHEFF_MASK', '>i4'), ('SIM_REDSHIFT_HELIO', '>f4'), ('SIM_REDSHIFT_CMB', '>f4'), ('SIM_REDSHIFT_HOST', '>f4'), ('SIM_REDSHIFT_FLAG', '>i2'), ('SIM_VPEC', '>f4'), ('SIM_DLMU', '>f4'), ('SIM_LENSDMU', '>f4'), ('SIM_RA', '>f8'), ('SIM_DEC', '>f8'), ('SIM_MWEBV', '>f4'), ('SIM_PEAKMJD', '>f4'), ('SIM_MAGSMEAR_COH', '>f4'), ('SIM_AV', '>f4'), ('SIM_RV', '>f4'), ('SIMSED_SALT2x0', '>f4'), ('SIMSED_PARAM(stretch)', '>f4'), ('SIMSED_PARAM(color)', '>f4'), ('SIM_PEAKMAG_g', '>f4'), ('SIM_PEAKMAG_r', '>f4'), ('SIM_EXPOSURE_g', '>f4'), ('SIM_EXPOSURE_r', '>f4'), ('SIM_SUBSAMPLE_INDEX', '>i2')]))
