# MAKE DATA
# orig_lc, timesX, y_predict = classify_lasair_light_curves(object_names=[
#     'ZTF18abxftqm',  # TDE
#     'ZTF19aadnmgf',  # SNIa
#     'ZTF18acmzpbf',  # SNIa
# ])
#
# with open('paper_plot_real_data.pickle', 'wb') as f:
#     pickle.dump([orig_lc, timesX, y_predict], f)

# PLOT FOR PAPER
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator

font = {'family': 'normal',
        'size': 60}
matplotlib.rc('font', **font)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

with open('paper_plot_real_data.pickle', 'rb') as f:
    orig_lc, timesX, y_predict = pickle.load(f)

CLASS_NAMES = ['Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'point-Ia', 'Kilonova',
               'SLSN-I',
               'PISN', 'ILOT', 'CART', 'TDE']
CLASS_COLOR = {'Pre-explosion': 'grey', 'SNIa-norm': 'tab:green', 'SNIbc': 'tab:orange', 'SNII': 'tab:blue',
               'SNIa-91bg': 'tab:red', 'SNIa-x': 'tab:purple', 'point-Ia': 'tab:brown', 'Kilonova': '#aaffc3',
               'SLSN-I': 'tab:olive', 'PISN': 'tab:cyan', 'ILOT': '#FF1493', 'CART': 'navy', 'TDE': 'tab:pink'}
PB_COLOR = {'u': 'tab:blue', 'g': 'tab:blue', 'r': 'tab:orange', 'i': 'm', 'z': 'k', 'Y': 'y'}
PB_MARKER = {'g': 'o', 'r': 's'}
PB_ALPHA = {'g': 0.3, 'r': 1.}

passbands = ['g', 'r']
use_interp_flux = False
step = True
texts = ['TDE ($z = 0.09$)', 'SNIa ($z = 0.08$)', 'SNIa ($z = 0.036$)']
names = ['ZTF18abxftqm', 'ZTF19aadnmgf', 'ZTF18acmzpbf']

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(45, 25), num="ZTF_real_data_examples", sharex='col')

for idx in range(len(orig_lc)):
    argmax = timesX[idx].argmax() + 1
    # ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
    # ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)

    for pb in passbands:
        if pb in orig_lc[idx].keys():
            ax[0][idx].errorbar(orig_lc[idx][pb]['time'], orig_lc[idx][pb]['flux'],
                                yerr=orig_lc[idx][pb]['fluxErr'], fmt=PB_MARKER[pb], label=pb,
                                c=PB_COLOR[pb], lw=4, markersize=14)

    new_t = np.array([orig_lc[idx][pb]['time'].values for pb in passbands]).flatten()
    new_t = np.sort(new_t[~np.isnan(new_t)])
    if not use_interp_flux:
        new_y_predict = []
        for classnum, classname in enumerate(CLASS_NAMES):
            new_y_predict.append(
                np.interp(new_t, timesX[idx][:argmax], y_predict[idx][:, classnum][:argmax]))

    class_lines = []
    for classnum, classname in enumerate(CLASS_NAMES):
        if not use_interp_flux:
            if step:
                class_lines.append(ax[1][idx].step(new_t, new_y_predict[classnum], '-', label=classname, color=CLASS_COLOR[classname], linewidth=4, where='post'))
            else:
                class_lines.append(ax[1][idx].plot(new_t, new_y_predict[classnum], '-', label=classname, color=CLASS_COLOR[classname], linewidth=4))
        else:
            class_lines.append(ax[1][idx].plot(timesX[idx][:argmax], y_predict[idx][:, classnum][:argmax], '-', label=classname, color=CLASS_COLOR[classname], linewidth=4))
    ax[0][idx].legend(frameon=True, fontsize=50, loc='upper right')
    # ax[1][idx].legend(frameon=True, fontsize=21.5)  # , loc='center right')  # , ncol=2)
    ax[1][idx].set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
    ax[0][idx].set_ylabel("Relative Flux")  # , fontsize=15)
    ax[1][idx].set_ylabel("Class Probability")  # , fontsize=18)
    # ax1.set_ylim(-0.1, 1.1)
    ax[1][idx].set_ylim(0, 1)
    ax[0][idx].set_xlim(left=min(new_t))  # ax1.set_xlim(-70, 80)
    # ax1.grid(True)
    # ax2.grid(True)
    ttl = ax[0][idx].set_title(names[idx], fontsize=60)
    ttl.set_position([.5, 1.02])
    plt.setp(ax[0][idx].get_xticklabels(), visible=False)
    ax[1][idx].yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added


handles, labels = ax[1][0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0., 0.0, 1., .02), borderaxespad=0.) #, mode='expand')
savename = 'ZTF_real_data_examples'
plt.tight_layout()
fig.subplots_adjust(hspace=0)
fig.subplots_adjust(bottom=0.24)
plt.savefig("{}.pdf".format(savename), bbox_extra_artists=(lgd,), bbox_inches="tight")
