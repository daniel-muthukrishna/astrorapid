"""
Example usage:
nice -n 19 python -m astrorapid.read_from_database.read_light_curves_from_database --offset 0 --offsetnext 70 --nprocesses 8 --savename 'testing' --combinefiles
"""

import os
import numpy as np
import h5py
import multiprocessing as mp
import pandas as pd
import argparse

from astrorapid.read_from_database.get_data import GetData
from astrorapid.process_light_curves import InputLightCurve


def read_light_curves_from_sql_database(data_release, fname, field_in='%', model_in='%', batch_size=100, offset=0,
                                        sort=True, passbands=('g', 'r'), known_redshift=True):
    print(fname)

    extrasql = ''  # "AND (objid LIKE '%00' OR objid LIKE '%50' OR sim_type_index IN (51,61,62,63,64,84,90,91,93))"  # ''#AND sim_redshift_host < 0.5 AND sim_peakmag_r < 23'
    getter = GetData(data_release)
    result = getter.get_lcs_data(
        columns=['objid', 'ptrobs_min', 'ptrobs_max', 'sim_peakmag_r', 'sim_redshift_host', 'mwebv', 'sim_dlmu',
                 'peakmjd', 'mwebv', 'ra', 'decl', 'hostgal_photoz', 'hostgal_photoz_err'],
        field=field_in, model=model_in, snid='%', limit=batch_size, offset=offset, shuffle=False, sort=sort,
        extrasql=extrasql)

    store = pd.HDFStore(fname)

    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, peakmag, redshift, mwebv, dlmu, peakmjd, mwebv, ra, dec, photoz, photozerr = head

        field, model, base, snid = objid.split('_')

        lc = getter.convert_pandas_lc_to_recarray_lc(phot, passbands=passbands)

        inputlightcurve = InputLightCurve(lc['mjd'], lc['flux'], lc['dflux'], lc['pb'], lc['zpt'], lc['photflag'], ra,
                                          dec, objid, redshift, mwebv, known_redshift=known_redshift,
                                          training_set_parameters={'class_number': int(model), 'peakmjd': peakmjd})

        savepd = inputlightcurve.preprocess_light_curve()
        store.append(objid, savepd)

    store.close()
    print("saved %s" % fname)


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
    data_release, i, save_dir, field_in, model_in, batch_size, sort, passbands, known_redshift = args
    offset = batch_size * i
    fname = os.path.join(save_dir, 'lc_{}.hdf5'.format(i))
    read_light_curves_from_sql_database(data_release=data_release, fname=fname, field_in=field_in, model_in=model_in,
                                        batch_size=batch_size, offset=offset, sort=sort, passbands=passbands,
                                        known_redshift=known_redshift)


def main():
    passbands = ('g', 'r')
    data_release = 'ZTF_20180716'
    field = 'MSIP'
    model = '%'
    known_redshift = True

    # Get number of objects
    # extrasql = ''
    # getter = GetData(data_release)
    # nobjects = next(getter.get_lcs_headers(field=field, model=model, get_num_lightcurves=True, big=False, extrasql=extrasql))
    # print("{} objects for model {} in field {}".format(nobjects, model, field))

    batch_size = 1000
    sort = True

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
        offset_next = 2200
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
            print(os.path.join(save_dir, 'lc_{}.hdf5'.format(i)))
            args_list.append((data_release, i, save_dir, field, model, batch_size, sort, passbands, known_redshift))

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
