import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import emcee
from scipy.stats import chisquare
import pylab
from collections import OrderedDict
from matplotlib import rc
import matplotlib as mpl
# mpl.use('GTK3AGG')
rc('text', usetex=False)
# rc('axes', unicode_minus=True)
# rc('font', **{'family':'serif','serif':['Times New Roman']})
from . import model_early_lightcurve

class EarlyMixin(object):
    """
    Methods to derive early classification features for LAobjects
    """

    def get_early_rise_rate(self, recompute=False, plot=False, passbands=('u', 'g', 'r', 'i', 'z', 'Y')):
        """
        Compute the early rise rate (slope) of the light curve.
        """

        earlyriserate = getattr(self, 'earlyriserate', None)
        a_fit = getattr(self, 'a_fit', None)
        c_fit = getattr(self, 'c_fit', None)
        if earlyriserate is not None:
            if not recompute:
                return earlyriserate, a_fit, c_fit

        earlyriserate = {}
        a_fit = {}
        c_fit = {}
        return_vals = {}
        outlc = self.get_lc(recompute=recompute)

        fit_func = getattr(self, 'early_fit_func', None)
        parameters = getattr(self, 'early_parameters', None)
        if fit_func is None or parameters is None:
            fit_func, parameters = model_early_lightcurve.fit_early_lightcurve(outlc)
            self.early_fit_func, self.early_parameters = fit_func, parameters

        if plot:
            plt.figure(figsize=(12, 10))
            ymin, ymax = -10, 10

        col = {'u': 'b', 'g': 'g', 'r': 'r', 'i': 'm', 'z': 'k', 'Y': 'y'}
        for i, pb in enumerate(passbands):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            tFluxUnred = tFluxUnred - np.median(tFluxUnred[ttime < 0])
            # mask = ttime > -30
            # tFluxUnred = tFluxUnred[mask]
            # ttime = ttime[mask]
            # tFluxErrUnred = tFluxErrUnred[mask]

            if len(ttime) <= 1 or not np.any(ttime < 0):
                return_vals[pb] = (-99, -99, -99)
                if plot:
                    plt.errorbar(ttime, tFluxUnred, yerr=tFluxErrUnred, fmt='.', color=col[pb], label=pb)
                continue

            fit_flux = fit_func(np.arange(min(ttime), max(ttime), 0.2), *parameters[pb])

            a, c, t0 = parameters[pb]

            t1 = 1 + t0
            t2 = 10 + t0
            f1 = fit_func(t1, *parameters[pb])
            f2 = fit_func(t2, *parameters[pb])
            tearlyriserate = (f2 - f1) / (t2 - t1)

            earlyriserate[pb] = tearlyriserate
            a_fit[pb] = parameters[pb][0]
            c_fit[pb] = parameters[pb][1]
            return_vals[pb] = (tearlyriserate, parameters[pb][0], parameters[pb][1])

            if plot:
                label = "{:>2}: a ={:6.2f}, c ={:6.2f}, slope ={:6.2f}".format(pb, a, c, tearlyriserate)
                plt.errorbar(ttime, tFluxUnred, yerr=tFluxErrUnred, fmt='.', color=col[pb], label=label)
                plt.plot(np.arange(min(ttime), max(ttime), 0.2), fit_flux, color=col[pb])
                ymin = min(ymin, (min(tFluxUnred[ttime >= -40]) - max(tFluxErrUnred[ttime >= -40])))
                ymax = max(ymax, (max(tFluxUnred[ttime >= -40]) + max(tFluxErrUnred[ttime >= -40])))

        if plot:
            plt.title(self.objectId)
            plt.xlabel("Days since trigger (rest frame)", fontsize=13)
            plt.ylabel("Relative Flux (distance corrected)", fontsize=13)
            plt.xlim(-40, 15)
            plt.ylim(ymin, ymax)
            plt.autoscale(enable=True, axis='y', tight=True)
            plt.tick_params(axis='both', labelsize=12)

            plt.errorbar([0], [0], yerr=[0], alpha=0.0, label="$t_0$ ={:7.2f}".format(t0))
            plt.errorbar([0], [0], yerr=[0], alpha=0.0, label="$z = {:.3f}$".format(self.z))
            # plt.autoscale(enable=True, axis='y', tight=True)
            plt.legend(frameon=False, loc='upper left')
            plt.savefig(self.objectId + "_ylims_fit.pdf")
            plt.show()

        self.earlyriserate = earlyriserate
        self.a_fit = a_fit
        self.c_fit = c_fit
        return return_vals

    def get_color_at_n_days(self, n, recompute=True, passbands=('u', 'g', 'r', 'i', 'z', 'Y')):
        """
        Compute the colors at n days and the linear slope of the color
        passbands = ('u', 'g', 'r', 'i', 'z', 'Y')  # order of filters matters as it must be 'u-g' rather than 'g-u'
        """
        color = getattr(self, 'color', None)
        if color is not None:
            if not recompute:
                return color

        color = {}
        color_slope = {}
        outlc = self.get_lc(recompute=recompute)


        fit_func = getattr(self, 'early_fit_func', None)
        parameters = getattr(self, 'early_parameters', None)
        if fit_func is None or parameters is None:
            fit_func, parameters = model_early_lightcurve.fit_early_lightcurve(outlc)
            self.early_fit_func, self.early_parameters = fit_func, parameters

        ignorepb = []
        tflux_ndays = {}
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            if len(ttime) <= 1 or not np.any(ttime < 0):
                tflux_ndays[pb] = 0
                continue

            # Check if there is data after t0 trigger for this passband
            a, c, t0 = parameters[pb]
            if max(ttime) < t0:
                print('No data for', pb, ttime)
                ignorepb.append(pb)
            else:
                print("time", pb, ttime)

            n = t0 + n

            tflux_ndays[pb] = fit_func(n, *parameters[pb])

        # plt.figure()

        for i, pb1 in enumerate(passbands):
            for j, pb2 in enumerate(passbands):
                if i < j:
                    c = pb1 + '-' + pb2
                    if pb1 not in parameters.keys() or pb2 not in parameters.keys() or tflux_ndays[pb2] == 0 or (pb1 in ignorepb or pb2 in ignorepb):
                        color[c] = -99.
                        color_slope[c] = -99.
                        # print("Not plotting", c)
                        continue
                    color[c] = -2.5*np.log10(tflux_ndays[pb1] / tflux_ndays[pb2])
                    color_slope[c] = color[c] / n
                    # fit_t = np.arange(-30, 15, 0.2)
                    # plotcolor = -2.5*np.log10(fit_func(fit_t, *parameters[pb1])/fit_func(fit_t, *parameters[pb2]))
                    # plt.plot(fit_t[fit_t >= t0], plotcolor[fit_t >= t0], label=c)

        # # plt.title(self.objectId)
        # plt.xlabel("Days since trigger (rest frame)")
        # plt.ylabel("Color")
        # # plt.xlim(-40, 15)
        # plt.legend(frameon=False)
        # plt.show()
        self.color = color
        self.color_slope = color_slope
        return color, color_slope


def lnlike(params, times, fluxes, fluxerrs):
        # print(params)
        t0 = params[0]
        pars = params[1:]

        chi2 = 0
        for i, pb in enumerate(times):
            a, c = pars[i * 2:i * 2 + 2]

            model = np.heaviside((times[pb] - t0), 1) * (a * (times[pb] - t0) ** 2) + c
            chi2 += sum((fluxes[pb] - model) ** 2 / fluxerrs[pb] ** 2)

            # print(pb, a, c)

        # print('chi2', chi2, params)
        return -chi2


def lnprior(params):
    # print('params', params)
    t0 = params[0]
    pars = params[1:]

    if -35 < t0 < 0:
        return 0.0
    return -np.inf


def lnprob(params, t, flux, fluxerr):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, t, flux, fluxerr)


def emcee_fit_all_pb_lightcurves(times, fluxes, fluxerrs, ndim, x0=None, bounds=None):
    nwalkers = 200
    nsteps = 700
    burn = 50
    pos = np.array([x0 + 3*np.random.randn(ndim) for i in range(nwalkers)])
    # print(pos[:,0])

    # print("DATA IS TO FOLLOW")
    # print("times")
    # print(times)
    # print("fluxes")
    # print(fluxes)
    # print("fluxerrs")
    # print(fluxerrs)

    print("running mcmc...")

    # Ensure intial params within parameter bounds
    params = OrderedDict()
    params['t0'] = {'bounds': bounds[0], 'value': x0[0], 'scale': 1}
    i = 1
    for pb in times.keys():
        for name in ['a', 'c']:
            params[pb + ': ' + name] = {'bounds': bounds[i], 'value': x0[i], 'scale': 3}
            i += 1
    for i, name in enumerate(params.keys()):
        # import pdb; pdb.set_trace()
        lb, ub = params[name]['bounds']
        p0     = params[name]['value']
        std    = params[name]['scale']
        # take a 5 sigma range
        lr, ur = (p0-5.*std, p0+5.*std)
        ll = max(lb, lr, 0.)
        ul = min(ub, ur)
        ind = np.where((pos[:,i] <= ll) | (pos[:,i] >= ul))
        nreplace = len(pos[:,i][ind])
        pos[:,i][ind] = np.random.rand(nreplace)*(ul - ll) + ll
    # print('new', np.array(pos)[:,0])
    # print(pos)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(times, fluxes, fluxerrs))

    # FInd parameters of lowest chi2
    pos, prob, state = sampler.run_mcmc(pos, 1)
    opos, oprob, orstate = [], [], []
    for pos, prob, rstate in sampler.sample(pos, prob, state, iterations=nsteps):
        opos.append(pos.copy())
        oprob.append(prob.copy())
    pos = np.array(opos)
    prob = np.array(oprob)
    nstep, nwalk = np.unravel_index(prob.argmax(), prob.shape)
    bestpars = pos[nstep, nwalk]
    posterior=prob

    print("best", bestpars)
    # import pylab
    # for j in range(nwalkers):
    #     pylab.plot(posterior[:, j])
    # for i in range(len(params)):
    #     pylab.figure()
    #     pylab.title(list(params.keys())[i])
    #     for j in range(nwalkers):
    #         pylab.plot(pos[:, j, i])

    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    # samples[:, 2] = np.exp(samples[:, 2])
    # bestpars2 = list(map(lambda v: (v[0]), zip(*np.percentile(samples, [50], axis=0))))
    # print('b2', bestpars2)

    print(bestpars)
    t0 = bestpars[0]
    bestpars = bestpars[1:]

    # for i, pb in enumerate(times):
    #     if len(times[pb][times[pb] > t0]) == 0:
    #         bestpars[i * 2] = 0  # set `a' to zero if no data points above t0

    best = {pb: np.append(bestpars[i * 2:i * 2 + 2], t0) for i, pb in enumerate(times)}

    # to_delete = []
    # for i, name in enumerate(params):
    #     if not np.any(samples[:, i]):  # all zeros for parameter then delete column
    #         to_delete.append((i, name))
    # for i, name in to_delete:
    #     samples = np.delete(samples, i, axis=1)
    #     print(name)
    #     del params[name]
    #
    # np.save('samples', samples[::100])

    return best
