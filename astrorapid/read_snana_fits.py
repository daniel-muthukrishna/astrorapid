import numpy as np
import multiprocessing as mp
import pandas as pd
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


def read_fits_file(args):
    head_file, phot_file, passbands, known_redshift, calculate_t0 = args

    getter = GetData()

    light_curves = {}

    header_HDU = afits.open(head_file)
    header_data = header_HDU[1].data

    for i, head in enumerate(header_data):
        class_num = head['SIM_TYPE_INDEX']
        snid = head['SNID']
        objid = '{}_{}'.format(class_num, snid)
        ptrobs_min = head['PTROBS_MIN']
        ptrobs_max = head['PTROBS_MAX']
        redshift = head['SIM_REDSHIFT_HOST']
        peakmjd = head['PEAKMJD']
        mwebv = head['MWEBV']
        ra = head['RA']
        if 'DEC' in header_data.names:
            dec = head['DEC']
        else:
            dec = head['DECL']
        print(i, len(header_data))

        try:
            phot_data = getter.get_light_curve_array(phot_file, ptrobs_min, ptrobs_max)

            lc = getter.convert_pandas_lc_to_recarray_lc(phot_data, passbands=passbands)

            inputlightcurve = InputLightCurve(lc['mjd'], lc['flux'], lc['dflux'], lc['pb'], lc['photflag'], ra,
                                              dec, objid, redshift, mwebv, known_redshift=known_redshift,
                                              training_set_parameters={'class_number': class_num, 'peakmjd': peakmjd},
                                              calculate_t0=calculate_t0)
            light_curves[objid] = inputlightcurve.preprocess_light_curve()
        except IndexError as e:
            print("No detections:", e)  # TODO: maybe do better error checking in future
        except AttributeError as e:
            print("phot_data is NoneType", e)  # TODO: maybe fix this later - rare case
        except ValueError as e:
            print("MCMC error while fitting t0", e)
        except Exception as e:
            print("Unspecified error", e, "Ignoring light curve", objid)

    return light_curves


def read_light_curves_from_snana_fits_files(head_files, phot_files, passbands=('g', 'r'), known_redshift=True,
                                            nprocesses=1, calculate_t0=True):
    """ Save lightcurves from SNANA HEAD AND PHOT FITS files

    Parameters
    ----------
    head_files : list
        List of filepaths of all SNANA HEAD files.
    phot_files : list
        List of filepaths of all SNANA PHOT files.
    passbands : tuple
        passband filters.
    known_redshift : bool
        Whether to use redshift during training.
    nprocesses : int
        Number of processes to run multiprocessing on.
    calculate_t0 : bool
        Optional parameter. If this is False, t0 will not be computed.

    """

    args_list = []
    for i, head_file in enumerate(head_files):
        assert phot_files[i].split('_')[-2].split('-')[1] == head_files[i].split('_')[-2].split('-')[1]
        head_files_part = head_files[i]
        phot_files_part = phot_files[i]
        args_list.append((head_files_part, phot_files_part, passbands, known_redshift, calculate_t0))

    light_curves = {}
    if nprocesses == 1:
        for args in args_list:
            light_curves.update(read_fits_file(args))
    else:
        pool = mp.Pool(nprocesses)
        results = pool.map_async(read_fits_file, args_list)
        pool.close()
        pool.join()

        outputs = results.get()
        print('combining results...')
        for i, output in enumerate(outputs):
            print(i, len(outputs))
            light_curves.update(output)

    return light_curves

