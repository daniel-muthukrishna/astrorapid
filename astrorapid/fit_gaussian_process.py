import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import celerite
from celerite import terms
from scipy.optimize import minimize


def combined_neg_log_like(params, fluxes, gp_lcs, passbands):
    loglike = 0
    for pb in passbands:
        gp_lcs[pb].set_parameter_vector(params)
        y = fluxes[pb]
        loglike += gp_lcs[pb].log_likelihood(y)

    return -loglike


def fit_gaussian_process_one_argument(args):
    lc, objid, passbands, plot, extrapolate = args

    return fit_gaussian_process(lc, objid, passbands, plot, extrapolate)


def fit_gaussian_process(lc, objid, passbands, plot, extrapolate, bad_loglike_thresh=-2000):
    print(f"Fitting GP to {objid}")
    gp_lc = {}
    if plot:
        plt.figure()

    kernel = terms.Matern32Term(log_sigma=5., log_rho=3.)
    times, fluxes, fluxerrs = {}, {}, {}
    for pbidx, pb in enumerate(passbands):
        pbmask = lc['passband'] == pb

        sortedidx = np.argsort(lc[pbmask]['time'].data)
        times[pb] = lc[pbmask]['time'].data[sortedidx]
        fluxes[pb] = lc[pbmask]['flux'].data[sortedidx]
        fluxerrs[pb] = lc[pbmask]['fluxErr'].data[sortedidx]

        try:
            gp_lc[pb] = celerite.GP(kernel)
            gp_lc[pb].compute(times[pb], fluxerrs[pb])
        except Exception as e:
            print("Failed object", objid, e)
            return

        # print("Initial log likelihood: {0}".format(gp_lc[pb].log_likelihood(fluxes[pb])))
        initial_params = gp_lc[pb].get_parameter_vector()  # This should be the same across passbands
        bounds = gp_lc[pb].get_parameter_bounds()

    # Optimise parameters
    try:
        r = minimize(combined_neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds,
                     args=(fluxes, gp_lc, passbands))
        # print(r)
    except Exception as e:
        print("Failed object", objid, e)
        return

    for pbidx, pb in enumerate(passbands):
        gp_lc[pb].set_parameter_vector(r.x)
        time = times[pb]
        flux = fluxes[pb]
        fluxerr = fluxerrs[pb]

        # print("Final log likelihood: {0}".format(gp_lc[pb].log_likelihood(flux)))

        # Remove objects with bad fits
        if extrapolate is True:
            x = np.linspace(min(min(time), -70), max(max(time), 80), 5000)
        else:
            x = np.linspace(min(time), max(time), 5000)
        pred_mean, pred_var = gp_lc[pb].predict(flux, x, return_var=True)
        if np.any(~np.isfinite(pred_mean)) or gp_lc[pb].log_likelihood(flux) < bad_loglike_thresh:
            print("Bad fit for object", objid)
            return

        # Plot GP fit
        if plot:
            # Predict with GP
            if extrapolate:
                x = np.linspace(min(min(time), -70), max(max(time), 80), 5000)
            else:
                x = np.linspace(min(time), max(time), 5000)
            pred_mean, pred_var = gp_lc[pb].predict(flux, x, return_var=True)
            pred_std = np.sqrt(pred_var)

            color = {'g': 'tab:green', 'r': "tab:red", 'i': "tab:purple", 'z': "tab:brown"}
            # plt.plot(time, flux, "k", lw=1.5, alpha=0.3)
            plt.errorbar(time, flux, yerr=fluxerr, fmt=".", capsize=0, color=color[pb])
            plt.plot(x, pred_mean, color=color[pb], label=pb)
            plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color[pb], alpha=0.3,
                             edgecolor="none")

    if plot:
        plt.xlabel("Days since trigger", fontsize=15)
        plt.ylabel("Flux", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=15)
        plt.show()
        # if extrapolate:
        #     plt.savefig(f'/Users/danmuth/PycharmProjects/transomaly/plots/gp_fits/extrapolated/gp_{objid}.pdf')
        # else:
        #     plt.savefig(f'/Users/danmuth/PycharmProjects/transomaly/plots/gp_fits/gp_{objid}.pdf')
        plt.close()

    return gp_lc, objid


def save_gps(light_curves, save_dir='data/saved_light_curves/', class_num=None, passbands=('g', 'r'), plot=False,
             nprocesses=1, redo=False, extrapolate=False):
    """ Save gaussian process fits.
    Don't plot in parallel
    """
    if extrapolate:
        save_gp_filepath = os.path.join(save_dir, f"gp_classnum_{class_num}_extrapolate.pickle")
    else:
        save_gp_filepath = os.path.join(save_dir, f"gp_classnum_{class_num}.pickle")

    if os.path.exists(save_gp_filepath) and not redo:
        with open(save_gp_filepath, "rb") as fp:  # Unpickling
            saved_gp_fits = pickle.load(fp)
    else:
        args_list = []
        for objid, lc in light_curves.items():
            args_list.append((lc, objid, passbands, plot, extrapolate))

        saved_gp_fits = {}
        if nprocesses == 1:
            for args in args_list:
                out = fit_gaussian_process_one_argument(args)
                if out is not None:
                    gp_lc, objid = out
                    saved_gp_fits[objid] = gp_lc
        else:
            pool = mp.Pool(nprocesses)
            results = pool.map_async(fit_gaussian_process_one_argument, args_list)
            pool.close()
            pool.join()

            outputs = results.get()
            print('combining results...')
            for i, output in enumerate(outputs):
                print(i, len(outputs))
                if output is not None:
                    gp_lc, objid = output
                    saved_gp_fits[objid] = gp_lc

        with open(save_gp_filepath, "wb") as fp:  # Pickling
            pickle.dump(saved_gp_fits, fp)

    return saved_gp_fits
