import os
import astrorapid.get_training_data
import astrorapid.get_custom_data
from astrorapid.custom_classifier import create_custom_classifier


def main():
    """ Train Neural Network classifier """

    script_dir = os.path.dirname(os.path.abspath(__file__))

    create_custom_classifier(get_data_func=astrorapid.get_training_data.get_data_from_snana_fits,
                             data_dir=os.path.join(script_dir, '..', 'data/ZTF_20190512'),
                             class_nums=(1, 2, 12, 14, 3, 13, 41, 43, 51, 60, 61, 62, 63, 64, 70),
                             class_name_map={1: 'SNIa-norm', 2: 'SNII', 12: 'SNII', 14: 'SNII', 3: 'SNIbc', 13: 'SNIbc', 41: 'SNIa-91bg', 43: 'SNIa-x', 51: 'Kilonova', 60: 'SLSN-I', 61: 'PISN', 62: 'ILOT', 63: 'CART', 64: 'TDE', 70: 'AGN'},
                             reread_data=False,
                             train_size=0.6,
                             contextual_info=('redshift',),
                             passbands=('g', 'r'),
                             retrain_network=False,
                             train_epochs=100,
                             zcut=0.5,
                             bcut=True,
                             ignore_classes=(61, 62, 63, 70),
                             nprocesses=1,
                             nchunks=10000,
                             otherchange='',
                             training_set_dir=os.path.join(script_dir, '..', 'training_set_files'),
                             save_dir=os.path.join(script_dir, '..', 'data/saved_light_curves'),
                             fig_dir=os.path.join(script_dir, '..', 'training_set_files', 'Figures', 'ZTF_with_redshift_2LSTM100_w_dropout0.0_epochs_100'),
                             plot=True
                             )


if __name__ == '__main__':
    main()
