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
import matplotlib.animation as animation

font = {'family': 'normal',
        'size': 60}
matplotlib.rc('font', **font)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']

with open('/Users/danmuth/PycharmProjects/astrorapid/paper_plot_real_data_new2019.pickle', 'rb') as f:
    orig_lc, timesX, y_predict = pickle.load(f)
    y_predict = list(y_predict)
with open('/Users/danmuth/PycharmProjects/astrorapid/save_real_mags_new2019.pickle', 'rb') as f:
    real_names, real_mjds, real_passbands, real_mags, real_magerrs, real_zeropoints, real_photflags = pickle.load(f)

with open('../paper_plot_real_data2.pickle', 'rb') as f:
    out = pickle.load(f)
    orig_lc[2], timesX[2], y_predict[2] = [out[i][1] for i in range(len(out))]
with open('paper_plot_real_data.pickle', 'rb') as f:
    out = pickle.load(f)
    _, timesX[2], y_predict[2] = [out[i][1] for i in range(len(out))]
with open('/Users/danmuth/PycharmProjects/astrorapid/save_real_mags.pickle', 'rb') as f:
    out = pickle.load(f)
    real_names[2], real_mjds[2], real_passbands[2], real_mags[2], real_magerrs[2], real_zeropoints[2], real_photflags[2] = [out[i][1] for i in range(len(out))]

# mask = (y_predict > 0.1)
# y_predict[mask] = 0.1*np.random.random(np.shape(y_predict[mask])) + y_predict[mask]

CLASS_NAMES = ['Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'point-Ia', 'Kilonova',
               'SLSN-I', 'PISN', 'ILOT', 'CART', 'TDE']#, 'AGN']
CLASS_COLOR = {'Pre-explosion': 'grey', 'SNIa-norm': 'tab:green', 'SNIbc': 'tab:orange', 'SNII': 'tab:blue',
               'SNIa-91bg': 'tab:red', 'SNIa-x': 'tab:purple', 'point-Ia': 'tab:brown', 'Kilonova': '#aaffc3',
               'SLSN-I': 'tab:olive', 'PISN': 'tab:cyan', 'ILOT': '#FF1493', 'CART': 'navy', 'TDE': 'tab:pink',
               'AGN': 'bisque'}
PB_COLOR = {'u': 'tab:blue', 'g': 'tab:blue', 'r': 'tab:orange', 'i': 'm', 'z': 'k', 'Y': 'y'}
PB_MARKER = {'g': 'o', 'r': 's'}
PB_ALPHA = {'g': 0.3, 'r': 1.}

passbands = ['g', 'r']
use_interp_flux = False
step = True
texts = ['TDE ($z = 0.074$)', 'TDE ($z = 0.08$)', 'SNIa ($z = 0.037$)']
names = ['ZTF19aarioci', 'ZTF19aabbnzo', 'ZTF19aadnmgf']

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(45, 25), num="ZTF_real_data_examples", sharex='col')

for idx in range(len(orig_lc)):
    argmax = timesX[idx].argmax() + 1
    # ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
    # ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)

    for pbidx, pb in enumerate(passbands):
        if pb in orig_lc[idx].keys():
            # ax[0][idx].errorbar(orig_lc[idx][pb]['time'], orig_lc[idx][pb]['flux'],
            #                     yerr=orig_lc[idx][pb]['fluxErr'], fmt=PB_MARKER[pb], label=pb,
            #                     c=PB_COLOR[pb], lw=4, markersize=14)
            # ax[0][idx].errorbar(orig_lc2[idx][pb]['time'], orig_lc2[idx][pb]['flux'],
            #                     yerr=orig_lc2[idx][pb]['fluxErr'], fmt=PB_MARKER[pb], label=pb,
            #                     c=PB_COLOR[pb], lw=4, markersize=14)
            trigger_mjd = orig_lc[idx]['otherinfo'][0][3]
            pbmask = (real_passbands[idx] == pbidx + 1) & ((real_photflags[idx] != 0) | (real_mjds[idx] <= trigger_mjd))
            # ax[0][idx].errorbar(real_mjds[idx][pbmask]-trigger_mjd, real_mags[idx][pbmask],
            #                     yerr=real_magerrs[idx][pbmask], fmt=PB_MARKER[pb], label=pb,
            #                     c=PB_COLOR[pb], lw=4, markersize=14)
            ax[0][idx].errorbar(orig_lc[idx][pb]['time'].dropna(), real_mags[idx][pbmask],
                                yerr=real_magerrs[idx][pbmask], fmt=PB_MARKER[pb], label=pb,
                                c=PB_COLOR[pb], lw=4, markersize=14)
            # ax[0][idx].errorbar(orig_lc[idx][pb]['time'], -2.5*np.log10(orig_lc[idx][pb]['flux']) + 26.2,
            #                     yerr=2.5*orig_lc[idx][pb]['fluxErr']/orig_lc[idx][pb]['flux']/np.log(10.), fmt=PB_MARKER[pb], label=pb,
            #                     c=PB_COLOR[pb], lw=4, markersize=14)

    ax[0][idx].invert_yaxis()
    new_t = np.array([orig_lc[idx][pb]['time'].values for pb in passbands]).flatten()
    new_t = np.sort(new_t[~np.isnan(new_t)])
    if not use_interp_flux:
        new_y_predict = []
        for classnum, classname in enumerate(CLASS_NAMES[:13]):
            new_y_predict.append(
                np.interp(new_t, timesX[idx][:argmax], y_predict[idx][:, classnum][:argmax]))

    class_lines = []
    for classnum, classname in enumerate(CLASS_NAMES):
        if classname == 'ILOT':
            cname = 'FBOT'
        elif classname not in ('Pre-explosion', 'SNIa-norm', 'SNIbc', 'SNII', 'SNIa-91bg', 'SNIa-x', 'SLSN-I', 'ILOT', 'TDE'):
            continue
        else:
            cname = classname
        if not use_interp_flux:
            if step:
                class_lines.append(ax[1][idx].step(new_t, new_y_predict[classnum], '-', label=cname, color=CLASS_COLOR[classname], linewidth=4, where='post'))
            else:
                class_lines.append(ax[1][idx].plot(new_t, new_y_predict[classnum], '-', label=cname, color=CLASS_COLOR[classname], linewidth=4))
        else:
            class_lines.append(ax[1][idx].plot(timesX[idx][:argmax], y_predict[idx][:, classnum][:argmax], '-', label=cname, color=CLASS_COLOR[classname], linewidth=4))
    # ax[0][idx].legend(frameon=True, fontsize=50, loc='lower right')
    # ax[1][idx].legend(frameon=True, fontsize=21.5)  # , loc='center right')  # , ncol=2)
    ax[1][idx].set_xlabel("Days since trigger (rest frame)")  # , fontsize=18)
    ax[0][idx].set_ylabel("Magnitude")  # , fontsize=15)
    ax[1][idx].set_ylabel("Class Probability")  # , fontsize=18)
    # ax1.set_ylim(-0.1, 1.1)
    # ax[1][idx].set_ylim(0, 1)
    ax[0][idx].set_xlim(left=min(new_t), right=max(timesX[idx][:argmax]))  # ax1.set_xlim(-70, 80)
    # ax1.grid(True)
    # ax2.grid(True)
    ttl = ax[0][idx].set_title(names[idx], fontsize=60)
    ttl.set_position([.5, 1.02])
    plt.setp(ax[0][idx].get_xticklabels(), visible=False)
    ax[1][idx].yaxis.set_major_locator(MaxNLocator(nbins=6, prune='upper'))  # added

ax[0][0].legend(frameon=True, fontsize=50, loc='upper right')
ax[0][1].legend(frameon=True, fontsize=50, loc='upper right')
ax[0][2].legend(frameon=True, fontsize=50, loc='lower right')
ax[1][0].set_xlim(right=75)
ax[1][1].set_xlim(right=75)
ax[0][1].set_ylim(bottom=20.3)
# ax[1][0].set_ylabel('Class probability', labelpad=15)

handles, labels = ax[1][0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0., 0.0, 1., .02), borderaxespad=0.) #, mode='expand')
savename = 'ZTF_real_data_examples_new2019_2'
plt.tight_layout()
fig.subplots_adjust(hspace=0)
fig.subplots_adjust(bottom=0.2)
plt.savefig("{}.pdf".format(savename), bbox_extra_artists=(lgd,), bbox_inches="tight")
