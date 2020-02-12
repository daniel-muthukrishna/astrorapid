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
                             reread_data=False,
                             contextual_info=('redshift',),
                             passbands=('g', 'r'),
                             retrain_network=False,
                             train_epochs=50,
                             zcut=0.5,
                             bcut=True,
                             variablescut=True,
                             nprocesses=None,
                             nchunks=10000,
                             aggregate_classes=False,
                             otherchange='',
                             training_set_dir=os.path.join(script_dir, '..', 'training_set_files'),
                             save_dir=os.path.join(script_dir, '..', 'data/saved_light_curves'),
                             fig_dir=os.path.join(script_dir, '..', 'training_set_files', 'Figures', 'ZTF_{}'),
                             plot=True
                             )


if __name__ == '__main__':
    main()
