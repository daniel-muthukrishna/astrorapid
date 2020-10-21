import numpy as np
from collections import OrderedDict
import copy

try:
    import emcee
except ImportError:
    print("You will need to install 'emcee' if you wish to train your own classifier on new data.")


def fit_early_lightcurve(outlc, earlytime=10):
    """
    Return tsquarize fit to early light curve
    """

    def fit_all_pb_light_curves(params, times, fluxes, fluxerrs):
        print(params)
        t0 = params[0]
        pars = params[1:]

        chi2 = 0
        for i, pb in enumerate(times):
            a, c = pars[i * 2:i * 2 + 2]

            model = np.heaviside((times[pb] - t0), 1) * (a * (times[pb] - t0) ** 2) + c
            chi2 += sum((fluxes[pb] - model) ** 2 / fluxerrs[pb] ** 2)

        print(chi2)
        return chi2

    def fit_func(t, a, c, t0):
        return np.heaviside((t - t0), 1) * (a * (t - t0) ** 2) + c

    copy.deepcopy(outlc)
    times = OrderedDict()
    fluxes = OrderedDict()
    fluxerrs = OrderedDict()
    passbands = np.unique(outlc['passband'].data)
    for i, pb in enumerate(passbands):
        pbmask = outlc['passband'] == pb
        ttime = outlc[pbmask]['time'].data
        tflux = outlc[pbmask]['flux'].data
        tfluxerr = outlc[pbmask]['fluxErr'].data
        tphotflag = outlc[pbmask]['photflag'].data

        minfluxpb = tflux.min()
        maxfluxpb = tflux.max()
        norm = maxfluxpb - minfluxpb

        tfluxrenorm = (tflux - minfluxpb) / norm
        tfluxerrrenorm = tfluxerr / norm

        if len(ttime) <= 1 or not np.any(ttime < 0):
            tfluxrenorm = tfluxrenorm - min(tfluxrenorm)
        else:
            tfluxrenorm = tfluxrenorm - np.median(tfluxrenorm[ttime < 0])
        # mask = ttime > -30
        # tfluxrenorm = tfluxrenorm[mask]
        # ttime = ttime[mask]
        # tfluxerrrenorm = tfluxerrrenorm[mask]

        earlymask = ttime < earlytime
        times[pb] = ttime[earlymask]
        fluxes[pb] = tfluxrenorm[earlymask]
        fluxerrs[pb] = tfluxerrrenorm[earlymask]

    remove_pbs = []
    x0 = [-12]
    bounds = [(-25, 0)]
    for pb in fluxes:
        if fluxes[pb].size == 0:
            remove_pbs.append(pb)
        else:
            x0 += [np.mean(fluxes[pb]), np.median(fluxes[pb])]
            bounds += [(0, max(fluxes[pb])), (min(fluxes[pb]), max(fluxes[pb]))]

    for pb in remove_pbs:
        del times[pb]
        del fluxes[pb]
        del fluxerrs[pb]

    # optimise_result = minimize(fit_all_pb_light_curves, x0=x0, args=(times, fluxes, fluxerrs), bounds=bounds)
    # t0 = optimise_result.x[0]
    # bestpars = optimise_result.x[1:]
    # best = {pb: np.append(bestpars[i*2:i*2+2], t0) for i, pb in enumerate(times)}

    ndim = len(x0)

    best = emcee_fit_all_pb_lightcurves(times, fluxes, fluxerrs, ndim, np.array(x0), bounds)
    # best, covariance = curve_fit(fit_func, time, flux, sigma=fluxerr, p0=[max(flux), min(flux)])

    # best = fit_all_pb_lightcurves(time, flux, fluxerr)
    # print('best', best)

    return fit_func, best


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
    return -0.5*chi2


def lnprior(params):
    # print('params', params)
    t0 = params[0]
    pars = params[1:]

    for par in pars:
        if par > 1e3 or par < -1e3:
            return -np.inf

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
    burn = 5
    pos = np.array([x0 + (([3] + len(times.keys()) * [0.1, 0.05]) * np.random.randn(ndim)) for i in range(nwalkers)])
    # print(pos[:,0])

    # print("DATA IS TO FOLLOW")
    # print("times")
    # print(times)
    # print("fluxes")
    # print(fluxes
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
        p0 = params[name]['value']
        std = params[name]['scale']
        # take a 5 sigma range
        lr, ur = (p0 - 5. * std, p0 + 5. * std)
        ll = max(lb, lr, 0.)
        ul = min(ub, ur)
        ind = np.where((pos[:, i] <= ll) | (pos[:, i] >= ul))
        nreplace = len(pos[:, i][ind])
        pos[:, i][ind] = np.random.rand(nreplace) * (ul - ll) + ll
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
    posterior = prob

    # print("best", bestpars)
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

    # print(bestpars)
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
