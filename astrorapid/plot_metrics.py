import os
import numpy as np
import pandas as pd
from distutils.spawn import find_executable
from scipy.interpolate import interp1d
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

from astrorapid.classifier_metrics import plot_confusion_matrix, compute_multiclass_roc_auc, compute_precision_recall, \
    plasticc_log_loss

try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.ticker import MaxNLocator
    import matplotlib.animation as animation
    import imageio
    if find_executable('latex'):
        plt.rcParams['text.usetex'] = True
    plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

except ImportError:
    print("Warning: You will need to install 'matplotlib' and 'imageio' if you want to plot the "
          "classification performance metrics.")

COLORS = ['grey', 'tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:purple', 'tab:brown', '#aaffc3', 'tab:olive',
          'tab:cyan', '#FF1493', 'navy', 'tab:pink', 'lightcoral', '#228B22', '#aa6e28', '#FFA07A']
COLCLASS = {'Pre-explosion': 'grey', 'SNIa-norm': 'tab:green', 'SNIbc': 'tab:orange', 'SNII': 'tab:blue',
            'SNIa-91bg': 'tab:red', 'SNIa-x': 'tab:purple', 'point-Ia': 'tab:brown', 'Kilonova': '#aaffc3',
            'SLSN-I': 'tab:olive', 'PISN': 'tab:cyan', 'ILOT': '#FF1493', 'CART': 'navy', 'TDE': 'tab:pink',
            'AGN': 'bisque'}
COLPB = {'u': 'tab:blue', 'g': 'tab:blue', 'r': 'tab:orange', 'i': 'm', 'z': 'k', 'Y': 'y'}
MARKPB = {'g': 'o', 'r': 's', 'i': 'x'}
ALPHAPB = {'g': 0.3, 'r': 1., 'i': 0.7}
WLOGLOSS_WEIGHTS = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
MINTIME = -70
MAXTIME = 80

def plot_metrics(class_names, model, X_test, y_test, fig_dir, timesX_test=None, orig_lc_test=None, objids_test=None,
                 passbands=('g', 'r'), num_ex_vs_time=100, init_day_since_trigger=-25):
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    class_names = sorted(class_names)

    y_pred = model.predict(X_test)
    y_test_indexes = np.argmax(y_test, axis=-1)
    y_pred_indexes = np.argmax(y_pred, axis=-1)

    accuracy = len(np.where(y_pred_indexes == y_test_indexes)[0])
    print("Accuracy is: {}/{} = {}".format(accuracy, len(y_test_indexes.flatten()),
                                           accuracy / len(y_test_indexes.flatten())))

    if "Pre-explosion" not in class_names:
        class_names = ["Pre-explosion"] + class_names

    # Set trailing zeros to -200
    for i in range(timesX_test.shape[0]):
        timesX_test[i][:np.argmin(timesX_test[i])] = -200
        timesX_test[i][np.argmax(timesX_test[i]) + 1:] = -200

    for cname in class_names:
        dirname = os.path.join(fig_dir + '/lc_pred', cname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # Plot accuracy vs time per class
    font = {'family': 'normal',
            'size': 36}
    matplotlib.rc('font', **font)

    # Plot classification example vs time
    for idx in np.arange(0, num_ex_vs_time):
        true_class = int(max(y_test_indexes[idx]))
        # print(true_class)
        # if true_class != 1:
        #     continue
        print("Plotting example vs time number {}, id \"{}\"...".format(idx, objids_test[idx]))
        argmax = timesX_test[idx].argmax() + 1

        lc_data = orig_lc_test[idx]
        used_passbands = [pb for pb in passbands if pb in lc_data['passband']]

        new_t = np.concatenate([lc_data[lc_data['passband'] == pb]['time'].data for pb in used_passbands])
        new_t = np.sort(new_t[~np.isnan(new_t)])
        new_y_predict = []

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15), num="classification_vs_time_{}".format(idx),
                                       sharex=True)

        for pbidx, pb in enumerate(passbands):
            if pb not in used_passbands:
                continue
            pbmask = lc_data['passband'] == pb
            # masktime = (lc_data[pbmask]['time'] > MINTIME) & (lc_data[pbmask]['time'] < MAXTIME)
            ax1.errorbar(lc_data[pbmask]['time'], lc_data[pbmask]['flux'],
                         yerr=lc_data[pbmask]['fluxErr'], fmt='.', label=pb, c=COLPB[pb],
                         lw=3, markersize=10, alpha=0.2)
            ax1.plot(timesX_test[idx][:argmax], X_test[idx][:, pbidx][:argmax], c=COLPB[pb],
                     lw=3)  # , markersize=10, marker='.'

        true_class = int(max(y_test_indexes[idx]))
        ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)
        try:
            redshift = lc_data.meta['redshift']
            mwebv = lc_data.meta['mwebv']
            b = lc_data.meta['b']
            trigger_mjd = lc_data.meta['trigger_mjd']
            t0 = lc_data.meta['t0']
            peakmjd = lc_data.meta['peakmjd']

            if t0 != -99:
                ax1.axvline(x=t0, color='grey', linestyle='--', linewidth=2)
                ax2.axvline(x=t0, color='grey', linestyle='--', linewidth=2)
                ax1.annotate('$t_0 = {}$'.format(round(t0, 1)), xy=(t0, 1), xytext=(t0 - 33, 0.9*max(orig_lc_test[idx]['r']['flux'])), color='grey')

            ax1.axvline(x=peakmjd - trigger_mjd, color='k', linestyle=':', linewidth=1)
            ax2.axvline(x=peakmjd - trigger_mjd, color='k', linestyle=':', linewidth=1)
        except Exception as e:
            print(e)

        class_accuracies = [timesX_test[idx][:argmax]]

        #
        for classnum, classname in enumerate(class_names):
            new_y_predict.append(np.interp(new_t, timesX_test[idx][:argmax], y_pred[idx][:, classnum][:argmax]))

        for classnum, classname in enumerate(class_names):
            ax2.plot(timesX_test[idx][:argmax], y_pred[idx][:, classnum][:argmax], '-', label=classname,
                     color=COLORS[classnum], linewidth=3)
            # ax2.step(new_t, new_y_predict[classnum], '-', label=classname,
            #          color=COLORS[classnum], linewidth=3, where='post')
            class_accuracies.append(y_pred[idx][:, classnum][:argmax])
        ax1.legend(frameon=True, fontsize=33)
        ax2.legend(frameon=True, fontsize=20.5, ncol=1, loc='right')
        ax2.set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
        ax1.set_ylabel("Relative Flux")  # , fontsize=15)
        ax2.set_ylabel("Class Probability")  # , fontsize=18)
        # ax1.set_ylim(-0.1, 1.1)
        # ax2.set_ylim(0, 1)
        mintime_lc = min([min(lc_data[lc_data['passband'] == pb]['time']) for pb in used_passbands])
        maxtime_lc = max([max(lc_data[lc_data['passband'] == pb]['time']) for pb in used_passbands])
        ax1.set_xlim(max(mintime_lc, MINTIME), min(maxtime_lc, MAXTIME))
        # ax1.set_xlim(MINTIME, MAXTIME)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.savefig(os.path.join(fig_dir + '/lc_pred',
                                 "{}_{}_{}_{}_{}_matrix_input2.pdf".format(objids_test[idx], idx, class_names[true_class], redshift,
                                                                                 peakmjd - trigger_mjd)))
        plt.savefig(os.path.join(fig_dir + '/lc_pred', class_names[true_class],
                                 "{}_{}_{}_{}_{}_matrix_input2.pdf".format(objids_test[idx], idx, class_names[true_class], redshift,
                                                                                 peakmjd - trigger_mjd)))
        plt.close()

    # Plot animated classification example vs time
    for idx in []: #[869]: #[181, 409, 491, 508, 765, 1156, 1335, 1358, 1570]:  # np.arange(765, 766):  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 25, 30]:
        print("Plotting animation example vs time", idx)

        #
        new_t = np.concatenate([lc_data[lc_data['passband'] == pb]['time'].values for pb in used_passbands])
        new_t = np.sort(new_t[~np.isnan(new_t)])
        new_y_predict = []
        # all_flux = list(orig_lc_test[idx]['g']['flux']) + list(orig_lc_test[idx]['r']['flux'])

        # timestep = timesX_test[idx][1] - timesX_test[idx][0]
        argmax = timesX_test[idx].argmax() + 1
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15), num="animation_classification_vs_time_{}".format(idx), sharex=True)

        ax1.legend(frameon=True, fontsize=33)
        ax2.legend(frameon=True, fontsize=20, loc='center right')  # , ncol=2)
        ax2.set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
        ax1.set_ylabel("Relative Flux")  # , fontsize=15)
        ax2.set_ylabel("Class Probability")  # , fontsize=18)
        ax2.set_ylim(bottom=0, top=0.9)
        ax1.set_ylim(top=1.15*max([max(lc_data[lc_data['passband'] == pb]['flux']) for pb in used_passbands]), bottom=-2700)
        ax1.set_xlim(MINTIME, MAXTIME)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)

        true_class = int(max(y_test_indexes[idx]))

        redshift = lc_data.meta['redshift']
        mwebv = lc_data.meta['mwebv']
        b = lc_data.meta['b']
        trigger_mjd = lc_data.meta['trigger_mjd']
        t0 = lc_data.meta['t0']
        peakmjd = lc_data.meta['peakmjd']

        t0 = -7.3
        ax1.axvline(x=t0, color='grey', linestyle='--', linewidth=2)
        ax2.axvline(x=t0, color='grey', linestyle='--', linewidth=2)
        ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)
        # ax1.axvline(x=peakmjd - trigger_mjd, color='k', linestyle=':', linewidth=1)
        # ax2.axvline(x=peakmjd - trigger_mjd, color='k', linestyle=':', linewidth=1)
        ax1.annotate('$t_0 = {}$'.format(round(t0, 1)), xy=(t0, 1), xytext=(t0 - 30, 0.9*max(orig_lc_test[idx]['r']['flux'])), color='grey')

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, bitrate=1800)

        #
        for classnum, classname in enumerate(class_names):
            new_y_predict.append(np.interp(new_t, timesX_test[idx][:argmax], y_pred[idx][:, classnum][:argmax]))

        def animate(i):
            for pbidx, pb in enumerate(passbands):
                if pb not in used_passbands:
                    continue
                # ax1.plot(timesX_test[idx][:argmax][:int(i+1)], X_test[idx][:, pbidx][:argmax][:int(i+1)], label=pb, c=COLPB[pb], lw=3)#, markersize=10, marker='.')
                if i + 1 >= len(new_t):
                    break
                # If less than 0.4 day gap in times skip
                if (i + 1) < len(new_t) and (new_t[int(i + 1)] - new_t[int(i)]) < 0.4:
                    break

                dea = [lc_data[pbmask]['time'] < new_t[int(i + 1)]]

                ax1.errorbar(np.array(lc_data[pbmask]['time'])[dea], np.array(lc_data[pbmask]['flux'])[dea],
                             yerr=np.array(lc_data[pbmask]['fluxErr'])[dea], fmt='.', label=pb,
                             c=COLPB[pb], lw=3, markersize=10)

            for classnum, classname in enumerate(class_names):
                # ax2.plot(timesX_test[idx][:argmax][:int(i+1)], y_pred[idx][:, classnum][:argmax][:int(i+1)], '-', label=classname, color=COLORS[classnum], linewidth=3)
                ax2.step(new_t[:int(i + 1)], new_y_predict[classnum][:int(i + 1)], '-', label=classname,
                         color=COLCLASS[classname], linewidth=3, where='post')

            # Don't repeat legend items
            ax1.legend(frameon=True, fontsize=33)
            ax2.legend(frameon=True, fontsize=20, loc='center right')
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            by_label1 = OrderedDict(zip(labels1, handles1))
            by_label2 = OrderedDict(zip(labels2, handles2))
            ax1.legend(by_label1.values(), by_label1.keys(), frameon=True, fontsize=33)
            ax2.legend(by_label2.values(), by_label2.keys(), frameon=True, fontsize=20, loc='center right')

        ani = animation.FuncAnimation(fig, animate, frames=len(new_t), repeat=True)
        ani.save(os.path.join(fig_dir + '/lc_pred', "classification_vs_time_{}_{}_{}_{}.mp4".format(idx, class_names[true_class], redshift, peakmjd - trigger_mjd)), writer=writer)

    print("Plotting Accuracy vs time per class...")
    fig = plt.figure("accuracy_vs_time_perclass", figsize=(13, 12))
    # fig = plt.figure(figsize=(13, 12))
    for classnum, classname in enumerate(class_names):
        correct_predictions_inclass = (y_test_indexes == classnum) & (y_pred_indexes == y_test_indexes)
        if not np.any(correct_predictions_inclass):
            print("There are no correct predictions for class \"{}\".".format(classname))
            continue
        time_bins = np.arange(-150, 150, 3.)

        times_binned_indexes = np.digitize(timesX_test, bins=time_bins, right=True)
        time_list_indexes_inclass, count_correct_vs_binned_time_inclass = np.unique(times_binned_indexes * correct_predictions_inclass, return_counts=True)
        time_list_indexes2_inclass, count_objects_vs_binned_time_inclass = np.unique(times_binned_indexes * (y_test_indexes == classnum), return_counts=True)

        time_list_indexes_inclass = time_list_indexes_inclass[time_list_indexes_inclass < len(time_bins)]
        count_correct_vs_binned_time_inclass = count_correct_vs_binned_time_inclass[time_list_indexes_inclass < len(time_bins)]
        time_list_indexes2_inclass = time_list_indexes2_inclass[time_list_indexes2_inclass < len(time_bins)]
        count_objects_vs_binned_time_inclass = count_objects_vs_binned_time_inclass[time_list_indexes2_inclass < len(time_bins)]

        start_time_index = int(np.where(time_list_indexes2_inclass == time_list_indexes_inclass[1])[0])
        end_time_index = int(np.where(time_list_indexes2_inclass == time_list_indexes_inclass[-1])[0])

        try:
            accuracy_vs_time_inclass = count_correct_vs_binned_time_inclass[1:] / count_objects_vs_binned_time_inclass[start_time_index:end_time_index+1]
        except Exception as e:
            print(e)
            continue

        try:
            assert np.all(time_list_indexes_inclass[1:] == time_list_indexes2_inclass[start_time_index:end_time_index+1])
        except Exception as e:
            print(e)
            pass

        plt.plot(time_bins[time_list_indexes_inclass[1:]], accuracy_vs_time_inclass, '-', label=classname, color=COLORS[classnum], lw=3)
    plt.xlim(left=init_day_since_trigger - 10, right=70)
    plt.xlabel("Days since trigger (rest frame)")
    plt.ylabel("Classification accuracy")
    plt.legend(frameon=True, fontsize=25, ncol=2, loc=0)  # , bbox_to_anchor=(0.05, -0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "accuracy_vs_time_perclass.pdf"), bbox_inches='tight')
    plt.close()

    font = {'family': 'normal',
            'size': 33}

    matplotlib.rc('font', **font)

    # Plot confusion matrix at different days past trigger
    print("Plotting Confusion Matrices...")
    time_bins = np.arange(-110, 110, 1.)
    nobjects = len(timesX_test)
    ntimesteps = len(time_bins)
    nclasses = y_test.shape[-1]
    y_test_indexes_days_past_trigger = np.zeros((nobjects, ntimesteps))
    y_pred_indexes_days_past_trigger = np.zeros((nobjects, ntimesteps))
    y_pred_days_past_trigger = np.zeros((nobjects, ntimesteps, nclasses))
    for objidx in range(nobjects):
        print(objidx, nobjects, 'For conf matrix') if objidx % 1000 == 0 else 0
        f = interp1d(timesX_test[objidx], y_test_indexes[objidx], kind='nearest', bounds_error=False,
                     fill_value='extrapolate')
        y_test_indexes_days_past_trigger[objidx][:] = f(time_bins)
        f = interp1d(timesX_test[objidx], y_pred_indexes[objidx], kind='nearest', bounds_error=False,
                     fill_value='extrapolate')
        y_pred_indexes_days_past_trigger[objidx][:] = f(time_bins)
        for classidx in range(nclasses):
            classprob = y_pred[objidx][:, classidx]
            mintimeidx = 0
            maxtimeidx = np.argmax(timesX_test[objidx])
            f = interp1d(timesX_test[objidx], classprob, kind='linear', bounds_error=False,
                         fill_value=(classprob[mintimeidx], classprob[maxtimeidx]))
            y_pred_days_past_trigger[objidx][:, classidx][:] = f(time_bins)

    images_cf, images_roc, images_pr = [], [], []
    roc_auc, pr_auc = {}, {}
    wlogloss = {}
    for i, days_since_trigger in enumerate(list(np.arange(init_day_since_trigger, 25)) + list(np.arange(25, 70, step=5))):  # [('early2days', 2), ('late40days', 40)]: # [-25, -20, -15, -10, -5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]:
        print("Plotting CF matrix", i, days_since_trigger, "days")
        index = np.where(time_bins == days_since_trigger)[0][0]
        y_test_on_day_i = y_test_indexes_days_past_trigger[:, index]
        y_pred_on_day_i = y_pred_indexes_days_past_trigger[:, index]
        y_pred_prob_on_day_i = y_pred_days_past_trigger[:, index]

        name = 'since_trigger_{}'.format(days_since_trigger)
        title = '{} days since trigger'.format(days_since_trigger)
        cnf_matrix = confusion_matrix(y_test_on_day_i, y_pred_on_day_i)
        print(name, cnf_matrix)
        figname_cf = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=title,
                                           fig_dir=fig_dir + '/cf_since_trigger', name=name)
        images_cf.append(imageio.imread(figname_cf))

        figname_roc, roc_auc[days_since_trigger] = compute_multiclass_roc_auc(class_names,
                                                                              to_categorical(y_test_on_day_i,
                                                                                             num_classes=nclasses),
                                                                              y_pred_prob_on_day_i, name=name,
                                                                              fig_dir=fig_dir + '/roc_since_trigger',
                                                                              title=title)
        images_roc.append(imageio.imread(figname_roc))

        figname_pr, pr_auc[days_since_trigger] = compute_precision_recall(class_names, to_categorical(y_test_on_day_i,
                                                                                                      num_classes=nclasses),
                                                                          y_pred_prob_on_day_i, name=name,
                                                                          fig_dir=fig_dir + '/pr_since_trigger',
                                                                          title=title)
        images_pr.append(imageio.imread(figname_pr))

        try:
            wlogloss[days_since_trigger] = plasticc_log_loss(to_categorical(y_test_on_day_i, num_classes=nclasses),
                                                             y_pred_prob_on_day_i, relative_class_weights=WLOGLOSS_WEIGHTS[:len(class_names)])
        except Exception as e:
            print("Cannot compute weighted PLAsTiCC Log Loss.", e)

        plt.close()

        objids_filename = os.path.join(fig_dir + '/truth_table_since_trigger', 'objids_{}.csv'.format(name))
        predicted_table_filename = os.path.join(fig_dir + '/truth_table_since_trigger',
                                                'predicted_prob_{}.csv'.format(name))
        truth_table_filename = os.path.join(fig_dir + '/truth_table_since_trigger', 'truth_table_{}.csv'.format(name))
        np.savetxt(objids_filename, objids_test, fmt='%s')
        np.savetxt(predicted_table_filename, y_pred_prob_on_day_i)
        np.savetxt(truth_table_filename, to_categorical(y_test_on_day_i, num_classes=nclasses))
        # out_objtable = os.path.join(fig_dir + '/truth_table_since_trigger', 'out_obj_table_{}.txt'.format(name))
        # out_truth = os.path.join(fig_dir + '/truth_table_since_trigger', 'out_truth_{}.csv'.format(name))
        # out_pred = os.path.join(fig_dir + '/truth_table_since_trigger', 'out_pred_{}.csv'.format(name))
        # make_tables(objids_filename, predicted_table_filename, truth_table_filename, directory='', processes=2, out_objtable=out_objtable, out_truth=out_truth, out_pred=out_pred)

    imageio.mimsave(os.path.join(fig_dir, 'animation_cf_since_trigger.gif'), images_cf, duration=0.25)
    imageio.mimsave(os.path.join(fig_dir, 'animation_roc_since_trigger.gif'), images_roc, duration=0.25)
    imageio.mimsave(os.path.join(fig_dir, 'animation_pr_since_trigger.gif'), images_pr, duration=0.25)

    font = {'family': 'normal',
            'size': 36}

    matplotlib.rc('font', **font)

    # Plot ROC AUC vs time
    plt.figure("ROC AUC vs time", figsize=(13, 12))
    roc_auc = pd.DataFrame(roc_auc).transpose()
    names = list(roc_auc.keys())

    plt.plot(list(roc_auc['micro'].index), list(roc_auc['micro'].values), color='navy', linestyle=':', linewidth=4,
             label='micro-average')
    for classnum, classname in enumerate(class_names):
        if classname not in names or classnum == 0:
            print(classname)
            continue
        times = list(roc_auc[classname].index)
        aucs = list(roc_auc[classname].values)
        plt.plot(times, aucs, '-', label=classname, color=COLORS[classnum], linewidth=4)

    plt.legend(frameon=True, fontsize=25)
    plt.xlabel("Days since trigger (rest frame)")  # , fontsize=18)
    plt.ylabel("AUC")  # , fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "auc_roc_vs_time.pdf"))
    plt.close()

    # # Plot Precision AUC vs time
    # plt.figure("ROC AUC vs time", figsize=(12, 11))
    # pr_auc = pd.DataFrame(pr_auc).transpose()
    # names = list(pr_auc.keys())
    # plt.plot(list(pr_auc['micro'].index), list(pr_auc['micro'].values), color='navy', linestyle=':', linewidth=4, label='micro')
    # for classnum, classname in enumerate(class_names):
    #     if classname not in names or classnum == 0:
    #         print(classname)
    #         continue
    #     times = list(pr_auc[classname].index)
    #     aucs = list(pr_auc[classname].values)
    #     plt.plot(times, aucs, '-', label=classname, color=COLORS[classnum], linewidth=2)
    #
    # plt.legend(frameon=False, fontsize=19)
    # plt.xlabel("Days since trigger")  # , fontsize=18)
    # plt.ylabel("AUC")  # , fontsize=15)
    # plt.savefig(os.path.join(fig_dir, "auc_pr_vs_time.pdf"))
    # plt.close()

    font = {'family': 'normal',
            'size': 36}
    matplotlib.rc('font', **font)

    # Plot weighted log loss vs time
    plt.figure("Weighted log loss vs time", figsize=(13, 12))
    plt.plot(list(wlogloss.keys()), list(wlogloss.values()), linewidth=4, label='Weighted Log loss')
    plt.xlabel("Days since trigger (rest frame)")  # , fontsize=18)
    plt.ylabel("Weighted log loss")  # , fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "wlogloss_vs_time.pdf"))
    plt.close()
    print(wlogloss)




    # for idx in [742]:#np.arange(0, 1000):
    #     true_class = int(max(y_test_indexes[idx]))
    #     print(true_class)
    #     if true_class != 1:
    #         continue
    #     print("Plotting example vs time", idx)
    #     argmax = timesX_test[idx].argmax() + 1
    #
    #     #
    #     new_t = np.array([orig_lc_test[idx][pb]['time'].values for pb in passbands]).flatten()
    #     new_t = np.sort(new_t[~np.isnan(new_t)])
    #     new_y_predict = []
    #
    #     fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15), num="classification_vs_time_{}".format(idx),
    #                                    sharex=True)
    #
    #     for pb in passbands:
    #         if pb in orig_lc_test[idx].keys():
    #             try:
    #                 ax1.errorbar(lc_data[pbmask]['time'], lc_data[pbmask]['flux'],
    #                              yerr=orig_lc_test[idx][pb]['fluxErr'], fmt='.', label=pb, c=COLPB[pb],
    #                              lw=3, markersize=10)
    #             except KeyError:
    #                 ax1.errorbar(orig_lc_test[idx][pb]['time'], orig_lc_test[idx][pb][5], yerr=orig_lc_test[idx][pb][6],
    #                              fmt='., label=pb, c=COLPB[pb], lw=3, markersize=10)
    #     true_class = int(max(y_test_indexes[idx]))
    #     ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
    #     ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)
    #     try:
    #         otherinfo = orig_lc_test[idx]['otherinfo'].values.flatten()
    #         redshift, b, mwebv, trigger_mjd, t0, peakmjd = otherinfo[0:6]
    #         t0=-7.3
    #         ax1.axvline(x=t0, color='grey', linestyle='--', linewidth=2)
    #         ax2.axvline(x=t0, color='grey', linestyle='--', linewidth=2)
    #         ax1.annotate('$t_0 = {}$'.format(round(t0, 1)), xy=(t0, 1), xytext=(t0 - 33, 0.9*max(orig_lc_test[idx]['r']['flux'])), color='grey')
    #         print(otherinfo[0:6])
    #         # ax1.axvline(x=peakmjd - trigger_mjd, color='k', linestyle=':', linewidth=1)
    #         # ax2.axvline(x=peakmjd - trigger_mjd, color='k', linestyle=':', linewidth=1)
    #     except Exception as e:
    #         print(e)
    #
    #     class_accuracies = [timesX_test[idx][:argmax]]
    #
    #     #
    #     for classnum, classname in enumerate(class_names):
    #         new_y_predict.append(np.interp(new_t, timesX_test[idx][:argmax], y_pred[idx][:, classnum][:argmax]))
    #
    #     for classnum, classname in enumerate(class_names):
    #         # ax2.plot(timesX_test[idx][:argmax], y_pred[idx][:, classnum][:argmax], '-', label=classname,
    #         #          color=COLORS[classnum], linewidth=3)
    #         ax2.step(new_t, new_y_predict[classnum], '-', label=classname,
    #                  color=COLORS[classnum], linewidth=3, where='post')
    #         class_accuracies.append(y_pred[idx][:, classnum][:argmax])
    #     ax1.legend(frameon=True, fontsize=33)
    #     ax2.legend(frameon=True, fontsize=20.5, ncol=1, loc='right')
    #     ax2.set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
    #     ax1.set_ylabel("Relative Flux")  # , fontsize=15)
    #     ax2.set_ylabel("Class Probability")  # , fontsize=18)
    #     # ax1.set_ylim(-0.1, 1.1)
    #     # ax2.set_ylim(0, 1)
    #     mintime_lc = min([min(orig_lc_test[idx][pb]['time']) for pb in passbands])
    #     maxtime_lc = max([max(orig_lc_test[idx][pb]['time']) for pb in passbands])
    #     ax1.set_xlim(max(mintime_lc, -70), min(maxtime_lc, 80))
    #     # ax1.set_xlim(-70, 80)
    #     plt.setp(ax1.get_xticklabels(), visible=False)
    #     ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added
    #     plt.tight_layout()
    #     fig.subplots_adjust(hspace=0)
    #     plt.savefig(os.path.join(fig_dir + '/lc_pred',
    #                              "classification_vs_time_{}_{}_{}_{}.pdf".format(idx, class_names[true_class], redshift,
    #                                                                              peakmjd - trigger_mjd)))
    #     plt.close()