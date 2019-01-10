import numpy as np

def calc_luminosity(flux, fluxerr, mu):
    """ Normalise flux light curves with distance modulus."""
    d = 10 ** (mu/5 + 1)
    dsquared = d**2

    norm = 1e18

    fluxout = flux * (4 * np.pi * dsquared/norm)
    fluxerrout = fluxerr * (4 * np.pi * dsquared/norm)

    return fluxout, fluxerrout