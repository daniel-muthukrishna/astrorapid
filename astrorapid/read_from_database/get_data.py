# -*- coding: UTF-8 -*-
"""
Get PLASTICC data from SQL database
"""
import sys
import os
import numpy as np
import warnings
import argparse
import pandas as pd
import astropy.table as at
import astropy.io.fits as afits
from collections import OrderedDict
from . import database
from astrorapid import helpers

ROOT_DIR = os.getenv('PLASTICC_DIR')
DATA_DIR = os.path.join(ROOT_DIR, 'plasticc_data')


def parse_getdata_options(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    def_sql_wildcard = '%'
    parser = argparse.ArgumentParser(description="Get options to the GetData structure")
    group = parser.add_mutually_exclusive_group(required=False)
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--data_release', required=True, help='PLAsTiCC data release index table to process')
    field_choices = ('WFD', 'DDF', '%')
    parser.add_argument('--field', required=False, default='DDF', type=str.upper, choices=field_choices, \
                        help='PLAsTiCC field to process')

    type_mapping = GetData.get_sntypes()
    inverse_mapping = {v: k for k, v in type_mapping.items()}

    model_choices = list(type_mapping.keys()).append('%')
    model_name_choices = list(type_mapping.values()).append('%')
    group.add_argument('--model', required=False, action="store", default=def_sql_wildcard, choices=model_choices, \
                       help='PLAsTiCC model to process')
    group.add_argument('--model_name', required=False, action="store", default=def_sql_wildcard,
                       choices=model_name_choices, \
                       help='PLAsTiCC model name to process')
    parser.add_argument('--base', required=False, default=def_sql_wildcard,
                        help='PLAsTiCC model base filename (probably not a good idea to touch this)')
    parser.add_argument('--snid', required=False, default=def_sql_wildcard,
                        help='PLAsTiCC object ID number (useful for debugging/testing)')
    parser.add_argument('--limit', required=False, type=int, default=None,
                        help='Limit the number of returned results from the MySQL index')
    parser.add_argument('--shuffle', required=False, type="bool", default="False",
                        help='Shuffle the returned results from the MySQL index')
    parser.add_argument('--sort', required=False, type="bool", default="True",
                        help='Sort the returned results from the MySQL index')
    parser.add_argument('--survey', required=False, default="LSST", 
                        help="Specify the survey to process")
    parser.add_argument('--offset', required=False, default=None, type=int,
                        help='Return the MySQL results AFTER offset rows')
    parser.add_argument('--extrasql', required=False, default=None,
                        help='Extra SQL for the selection function - enter as quoted string - used as is')
    args = parser.parse_args(args=argv)

    out = vars(args)
    model_name = out.pop('model_name')
    model = out.pop('model')
    if model_name == '%':
        if model == '%':
            out['model'] = '%'
        else:
            out['model'] = model
    else:
        this_model = inverse_mapping.get(model_name)
        if model == '%':
            out['model'] = this_model
        else:
            if this_model == model:
                out['model'] = this_model
            else:
                out['model'] = model
    return out


class GetData(object):
    """
    Class to access the ANTARES parsed PLaSTiCC index and light curve data
    """

    def __init__(self, data_release):
        self.data_release = "release_{}".format(data_release)
        self.phot_fields = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR', 'ZEROPT', 'PHOTFLAG']
        self.phot_fields_dtypes = {'FLT': np.str_, 'PHOTFLAG': np.int_}

    def get_phot_fields(self):
        """
        list of the photometry column names and a dictionary of NON-FLOAT
        columns

        For the default columns, this is only FLT
        """
        return list(self.phot_fields), dict(self.phot_fields_dtypes)

    def set_phot_fields(self, fields, dtypes):
        """
        set the list of photometry column fields to retrieve and a dictionary
        of data types for NON-FLOAT columns hashed by column name in the PHOT.FITS

        i.e. if your column is a float, don't even bother listing it

        Can be used to retrieve custom columns from the PHOT.FITS
        Kinda kludgey - se with caution
        """
        self.phot_fields = list(fields)
        self.phot_fields_dtypes = dict(dtypes)

    def get_object_ids(self):
        """ Get list of all object ids """
        obj_ids = database.exec_sql_query("SELECT objid FROM {0};".format(self.data_release))
        return obj_ids

    def get_column_for_sntype(self, column_name, sntype, field='%'):
        """ Get an sql column for a particular sntype class

        Parameters
        ----------
        column_name : str
            column name. E.g. column_name='peakmjd'
        sntype : int
            sntype number. E.g. sntype=4
        field : str, optional
            The field name. E.g. field='DDF' or field='WFD'. The default is '%' indicating that all fields will be included.

        Return
        -------
        column_out: list
            A list containing all the entire column for a particular sntype class
        """
        try:
            column_out = database.exec_sql_query(
                "SELECT {0} FROM {1} WHERE objid LIKE '{2}%' AND sntype={3};".format(column_name, self.data_release,
                                                                                     field, sntype))
            column_out = np.array(column_out)[:, 0]
        except IndexError:
            print("No data in the database satisfy the given arguments. field: {}, sntype: {}".format(field, sntype))
            return []
        return column_out

    def get_photfile_for_objid(self, objid):
        """
        Returns the phot file for the object ID
        """
        field, model, base, snid = objid.split('_')
        if field == 'IDEAL':
            filename = "{0}_MODEL{1}/{0}_{2}_PHOT.FITS".format(field, model, base)
        elif field == 'MSIP':
            filename = "ZTF_{0}_MODEL{1}/ZTF_{0}_{2}_PHOT.FITS".format(field, model, base)
        else:
            filename = "LSST_{0}_MODEL{1}/LSST_{0}_{2}_PHOT.FITS".format(field, model, base)
        phot_file = os.path.join(DATA_DIR, self.data_release.replace('release_', ''), filename)
        if not os.path.exists(phot_file):
            phot_file = phot_file + '.gz'
        return phot_file



    def get_light_curve(self, objid, ptrobs_min, ptrobs_max, standard_zpt=27.5):
        """ Get lightcurve from fits file

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
        phot_file = self.get_photfile_for_objid(objid)

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

    def get_light_curve_array(self, objid, ptrobs_min, ptrobs_max, standard_zpt=27.5):
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
        phot_file = self.get_photfile_for_objid(objid)

        try:
            phot_HDU = afits.open(phot_file, memmap=True)
        except Exception as e:
            message = f'Could not open photometry file {phot_file}'
            raise RuntimeError(message)

        phot_data = phot_HDU[1].data[ptrobs_min - 1:ptrobs_max]
        phot_data = at.Table(phot_data)
        return phot_data


    @staticmethod
    def convert_pandas_lc_to_recarray_lc(phot, passbands=('u', 'g', 'r', 'i', 'z', 'Y')):
        """
        ANTARES_object not Pandas format broken up by passband
        TODO: This is ugly - just have an option for get_lcs_data to return one or the other
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

    @staticmethod
    def get_sntypes():
        return helpers.get_sntypes()

    @staticmethod
    def aggregate_sntypes(reverse=False):
        return helpers.aggregate_sntypes(reverse=reverse)

    def get_avail_sntypes(self):
        """ Returns a list of the different transient classes in the database. """
        sntypes = database.exec_sql_query("SELECT DISTINCT sntype FROM {};".format(self.data_release))
        sntypes_map = self.get_sntypes()
        return sorted([sntype[0] for sntype in sntypes]), sntypes_map

    def get_lcs_headers(self, columns=None, field='%', model='%', base='%', snid='%', extrasql='', survey='LSST',
                        get_num_lightcurves=False, limit=None, shuffle=False, sort=True, offset=0, big=False):
        """ Gets the header data given specific conditions.

        Parameters
        ----------
        columns : list
            A list of strings of the names of the columns you want to retrieve from the database.
            You must at least include ['objid', 'ptrobs_min', 'ptrobs_max'] at the beginning of the input list.
            E.g. columns=['objid', 'ptrobs_min', 'ptrobs_max', 'sntype', 'peakmjd'].
        field : str, optional
            The field name. E.g. field='DDF' or field='WFD'. The default is '%' indicating that all fields will be included.
        model : str, optional
            The model number. E.g. model='04'. The default is '%' indicating that all model numbers will be included.
        base : str, optional
            The base name. E.g. base='NONIa'. The default is '%' indicating that all base names will be included.
        snid : str, optional
            The transient id. E.g. snid='87287'. The default is '%' indicating that all snids will be included.
        get_num_lightcurves : boolean, optional
            If this is True, then the return value is just a single iteration generator stating the number of
            light curves that satisfied the given conditions.
        limit : int, optional 
            Limit the results to this number (> 0)
        shuffle : bool, optional
            Randomize the order of the results - not allowed with `sort`
        sort : bool, optional
            Order the results by objid - overrides `shuffle` if both are set
        offset : int, optional
            Start returning MySQL results from this row number offset
        big : bool, optional
            If True, use a generator to retrieve results - cannot be used with
            get_num_lightcurves since generators have no length
        Return
        -------
        result: tuple or int
        """
        if columns is None:
            columns = ['objid', 'ptrobs_min', 'ptrobs_max']

        if get_num_lightcurves:
            columns = ['COUNT(objid)', ]
            big = False

        try:
            limit = int(limit)
            if limit <= 0:
                raise RuntimeError('prat')
        except Exception as e:
            limit = None

        try:
            offset = int(offset)
            if offset <= 0:
                raise RuntimeError('prat')
        except Exception as e:
            offset = None

        if limit is not None and shuffle is False and sort is False:
            sort = True

        extrasql_command = '' if extrasql is None else extrasql
        limit_command = '' if limit is None else " LIMIT {}".format(limit)
        offset_command = '' if offset is None else " OFFSET {}".format(offset)
        if model != '%':
            model = "{:02n}".format(int(model))

        if sort is True and shuffle is True:
            message = 'Cannot sort and shuffle at the same time! That makes no sense!'
            shuffle = False
            warnings.warn(message, RuntimeWarning)

        shuffle_command = '' if shuffle is False else " ORDER BY RAND()"
        sort_command = '' if sort is False else ' ORDER BY objid'
        extra_command = ''.join([extrasql_command, sort_command, shuffle_command, limit_command, offset_command])

        query = "SELECT {} FROM {} WHERE objid LIKE '{}_{}_{}_{}' {};".format(', '.join(columns), \
                                                                              self.data_release, field, model, base,
                                                                              snid, extra_command)
        if big:
            for result in database.exec_big_sql_query(query):
                yield result
        else:
            header = database.exec_sql_query(query)
            if get_num_lightcurves:
                num_lightcurves = int(header[0][0])
                yield num_lightcurves
            else:
                num_lightcurves = len(header)

            if num_lightcurves > 0:
                for result in header:
                    yield result
            else:
                print("No light curves in the database satisfy the given arguments. "
                      "field: {}, model: {}, base: {}, snid: {}".format(field, model, base, snid))
                return

    def get_lcs_data(self, columns=None, field='%', model='%', base='%', snid='%', survey='LSST',\
                     limit=None, shuffle=False, sort=True, offset=0, big=False, extrasql=''):
        """ Gets the light curve and header data given specific conditions. Returns a generator of LC info.

        Parameters
        ----------
        columns : list
            A list of strings of the names of the columns you want to retrieve from the database.
            You must at least include ['objid', 'ptrobs_min', 'ptrobs_max'] at the beginning of the input list.
            E.g. columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd'].
        field : str, optional
            The field name. E.g. field='DDF' or field='WFD'. The default is '%' indicating that all fields will be included.
        model : str, optional
            The model number. E.g. model='04'. The default is '%' indicating that all model numbers will be included.
        base : str, optional
            The base name. E.g. base='NONIa'. The default is '%' indicating that all base names will be included.
        snid : str, optional
            The transient id. E.g. snid='87287'. The default is '%' indicating that all snids will be included.
        limit : int, optional 
            Limit the results to this number (> 0)
        shuffle : bool, optional
            Randomize the order of the results - not allowed with `sort`
        sort : bool, optional
            Order the results by objid - overrides `shuffle` if both are set
        offset : int, optional
            Start returning MySQL results from this row number offset (> 0)
        big : bool, optional
            If True, use a generator to retrieve results - cannot be used with
            get_num_lightcurves since generators have no length
            

        Return
        -------
        result: tuple
            A generator tuple containing (objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, peak_mjd)
        phot_data : pandas DataFrame
            A generator containing a DataFrame with the MJD, FLT, MAG, MAGERR as rows and the the filter names as columns.
            E.g. Access the magnitude in the z filter with phot_data['z']['MAG'].
        """

        header = self.get_lcs_headers(columns=columns, field=field, \
                                      model=model, base=base, snid=snid, \
                                      limit=limit, sort=sort, shuffle=shuffle, offset=offset, \
                                      big=big, extrasql=extrasql)

        for h in header:
            objid, ptrobs_min, ptrobs_max = h[0:3]
            phot_data = self.get_light_curve(objid, ptrobs_min, ptrobs_max)
            yield h, phot_data

