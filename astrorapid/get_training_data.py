import os
import pickle
import matplotlib
matplotlib.use('Agg')

from astrorapid.read_snana_fits import read_light_curves_from_snana_fits_files


def get_data(class_num, data_dir='data/ZTF_20190512/', save_dir='data/saved_light_curves/', passbands=('g', 'r'),
             nprocesses=1, redo=False):
    save_lc_filepath = os.path.join(save_dir, f"lc_classnum_{class_num}.pickle")

    if os.path.exists(save_lc_filepath) and not redo:
        with open(save_lc_filepath, "rb") as fp:  # Unpickling
            light_curves = pickle.load(fp)
    else:
        class_dir = os.path.join(data_dir, 'ZTF_MSIP_MODEL{:02d}'.format(class_num))
        files = os.listdir(class_dir)

        head_files = []
        phot_files = []
        for file in files:
            filepath = os.path.join(data_dir, class_dir, file)
            if filepath.endswith('HEAD.FITS'):
                head_files.append(filepath)
                phot_files.append(filepath.replace('_HEAD.FITS', '_PHOT.FITS'))
            print(filepath)

        light_curves = read_light_curves_from_snana_fits_files(head_files, phot_files, passbands, known_redshift=False, nprocesses=nprocesses)

        with open(save_lc_filepath, "wb") as fp:  # Pickling
            pickle.dump(light_curves, fp)

    return light_curves
