import os
import astrorapid.get_training_data
import astrorapid.get_custom_data
from astrorapid.custom_classifier import create_custom_classifier


def main():
    """ Train Neural Network classifier """

    script_dir = os.path.dirname(os.path.abspath(__file__))

    create_custom_classifier(get_data_func=astrorapid.get_training_data.get_real_ztf_training_data,
                             data_dir=os.path.join(script_dir, '..', 'data/real_ZTF_data_from_osc'),
                             class_nums=('Ia', 'Ia91T', 'Ia91bg', 'Iapec', 'Iacsm', 'Iax',
                                         'II', 'IIP', 'IIL', 'IIpec', 'IIn', 'IIb', 'Ib', 'Ibn', 'Ic', 'IcBL', 'Ibc',
                                         'CC', 'SLSN', 'SLSNI', 'SLSNII'),
                             class_name_map={'Ia': 'SNIa', 'Ia91T': 'SNIa', 'Ia91bg': 'SNIa', 'Iapec': 'SNIa',
                                             'Iacsm': 'SNIa', 'Iax': 'SNIa', 'II': 'SNII', 'IIP': 'SNII', 'IIL':
                                                 'SNII', 'IIpec': 'SNII', 'IIn': 'SNII', 'IIb': 'SNII',
                                             'Ib': 'SNIbc', 'Ibn': 'SNIbc', 'Ic': 'SNIbc', 'IcBL': 'SNIbc',
                                             'Ibc': 'SNIbc', 'CC': 'CCSN',
                                             'SLSN': 'SLSN', 'SLSNI': 'SLSN', 'SLSNII': 'SLSN'},
                             reread_data=False,
                             train_size=0.6,
                             contextual_info=('redshift',),
                             passbands=('g', 'r'),
                             retrain_network=False,
                             train_epochs=100,
                             zcut=0.5,
                             bcut=True,
                             ignore_classes=('CC',),
                             nprocesses=None,
                             nchunks=10000,
                             otherchange='real-ztf-Ia-II-Ibc-SLSN',
                             training_set_dir=os.path.join(script_dir, '..', 'training_set_files'),
                             save_dir=os.path.join(script_dir, '..', 'data/saved_real_ZTF_light_curves'),
                             fig_dir=os.path.join(script_dir, '..', 'training_set_files', 'Figures', 'ZTF_real_data_with_redshift'),
                             plot=True
                             )


if __name__ == '__main__':
    main()
