import os
from astrorapid.prepare_training_set import PrepareTrainingSetArrays
from astrorapid.plot_metrics import plot_metrics
from astrorapid.neural_network_model import train_model
import astrorapid.get_custom_data


def create_custom_classifier(get_data_func, data_dir, class_nums=(1,2,), class_name_map=None, reread_data=False, train_size=0.6,
                             contextual_info=('redshift',), passbands=('g', 'r'), nobs=50, mintime=-70, maxtime=80,
                             timestep=3.0, retrain_network=False, train_epochs=50, dropout_rate=0,
                             train_batch_size=64, nunits=100, zcut=0.5, bcut=True,
                             ignore_classes=(), nprocesses=1, nchunks=1000, otherchange='',
                             training_set_dir='data/training_set_files', save_dir='data/saved_light_curves',
                             fig_dir='Figures', plot=True, num_ex_vs_time=100, init_day_since_trigger=-25,
                             augment_data=False, redo_processing=False):

    """
    Create a classifier with your own data and own training parameters.

    Parameters
    ----------
    get_data_func : func
        Function that reads your data and returns a light_curves object.
        It takes at least the arguments: class_num, data_dir, save_dir, passbands, known_redshift.
        See get_custom_data.py for an example template.
        E.g. get_data_func = get_data_custom(class_num, data_dir, save_dir, passbands)
    data_dir : str
        Directory where data is stored. E.g. data_dir='data/ZTF_20190512/'
    class_nums : tuple
        Class numbers (or names) to train on. E.g. class_nums=(1, 5, 6, 12, 41, 43, 51)
    class_name_map : dict or None
        This maps the class_nums onto class names.
        E.g. class_name_map = {1: 'SNIa-norm', 5: 'SNIbc', 6: 'SNII', 12: 'SNII', 41: 'SNIa-91bg', 43: 'SNIa-x', 51: 'Kilonova'}.
        You may use the same value for a different key if you want the classifier to join these two class_nums under the same label.
        If this is None, it will use the default mapping listed in get_sntypes in helpers.py.
    reread_data : bool
        If this is True, then it will reread_data your data and resave the processed files, otherwise
        it will check if the data has already been read, processed and saved.
    train_size : float
        Fraction of data to use for training, the remainder will be used for testing/validation.
        Usually choose a value between 0.5 - 1.0 depending on how much data you have.
    contextual_info : tuple of strings
        What contextual information to use while training. You can write any string in this tuple provided that it
        is stored as a keyword in the metadata of the light_curve Table object returned by the `get_data_func`.
    passbands : tuple of str
        Passbands to use.
        E.g. passbands=('g', 'r')
    nobs : int
        Number of points to use in interpolation of light curve between mintime and maxtime.
    mintime : int
        Days from trigger (minimum) to extract from light curve.
    maxtime : int
        Days from trigger (maximum) to extract from light curve.
    timestep : float
        Time-step between interpolated points in light curve.
    retrain_network : bool
        Whether to retrain the neural network or to use the saved network model.
    train_epochs : int
        Number of epochs to train the neural network.
        This is the number of times the nerual network sees each datum in the training set.
        The higher this is the better the classifier will find a local minimum, however, too high, and it might
        overfit and not generalise well to new data
    dropout_rate : float
        Value between 0.0 and 1.0 indicating fraction for dropout regularisation.
    train_batch_size : int
        Number of objects to use per step in gradient descent.
    units : int
        Number of LSTM units.
    zcut : float
        Do not train on objects with redshifts higher than zcut.
    bcut : bool
        If True, do not train on objects within 15 degrees of the galactic plane
    ignore_classes : tuple
        Will not train or test on classes listed in this tuple.
    nprocesses : int or None
        Number of computer processes to use while processing the data.
        If None, it will use all the available processors from os.cpu_count().
    nchunks : int
        Number of chunks to split the data set into before doing multiprocessing.
        This should be a small fraction of the number of total objects.
    otherchange : str
        A change in this text will signify that a change has been made to one of these training parameters
        and that the data should be resaved and the model retrained should resave the data and retrained.
    training_set_dir : str
        Name of directory to save the data that will be read by the neural network.
    save_dir : str
        Name of directory to save the processes data files.
    fig_dir : str
        Name of directory to save the Classifier metric figures such as confusion matrices.
    plot : bool
        Whether to plot classier metrics such as confusion matrices after training.
    num_ex_vs_time : int
        Number of example vs time light curves to plot.
    init_day_since_trigger : int
        Day since trigger from which to start plotting in vs time figures. Input a negative value for a day
        before trigger.
    augment_data : bool
        Whether to do Gaussian processing augmenting.
    redo_processing : bool
        Whether to redo processing AFTER reading data, saving GP fits and computing t0.
    """

    for dirname in [training_set_dir, data_dir, save_dir]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    for dirname in [fig_dir, fig_dir + '/cf_since_trigger', fig_dir + '/cf_since_t0', fig_dir + '/roc_since_trigger',
                fig_dir + '/lc_pred', fig_dir + '/pr_since_trigger', fig_dir + '/truth_table_since_trigger']:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # Prepare the training set, read the data files, and save processed files
    preparearrays = PrepareTrainingSetArrays(passbands, contextual_info, nobs, mintime, maxtime, timestep,
                                             reread_data, bcut, zcut, ignore_classes,
                                             class_name_map=class_name_map, nchunks=nchunks,
                                             training_set_dir=training_set_dir, data_dir=data_dir, save_dir=save_dir,
                                             get_data_func=get_data_func, augment_data=augment_data, redo_processing=redo_processing)
    X_train, X_test, y_train, y_test, labels_train, labels_test, class_names, class_weights, sample_weights, \
    timesX_train, timesX_test, orig_lc_train, orig_lc_test, objids_train, objids_test = \
        preparearrays.prepare_training_set_arrays(otherchange, class_nums, nprocesses, train_size)

    # Train the neural network model on saved files
    model = train_model(X_train, X_test, y_train, y_test, sample_weights=sample_weights, fig_dir=fig_dir,
                        retrain=retrain_network, epochs=train_epochs, plot_loss=plot, dropout_rate=dropout_rate,
                        batch_size=train_batch_size, nunits=nunits)

    # Plot classification metrics such as confusion matrices
    if plot:
        plot_metrics(class_names, model, X_test, y_test, fig_dir, timesX_test=timesX_test, orig_lc_test=orig_lc_test,
                     objids_test=objids_test, passbands=passbands, num_ex_vs_time=num_ex_vs_time, init_day_since_trigger=init_day_since_trigger)

    
