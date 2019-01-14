import numpy as np


def calc_luminosity(flux, fluxerr, mu):
    """ Normalise flux light curves with distance modulus.

    Parameters
    ----------
    flux : array
        List of floating point flux values.
    fluxerr : array
        List of floating point flux errors.
    mu : float
        Distance modulus from luminosity distance.

    Returns
    -------
    fluxout : array
        Same shape as input flux.
    fluxerrout : array
        Same shape as input fluxerr.

    """

    d = 10 ** (mu/5 + 1)
    dsquared = d**2

    norm = 1e18

    fluxout = flux * (4 * np.pi * dsquared/norm)
    fluxerrout = fluxerr * (4 * np.pi * dsquared/norm)

    return fluxout, fluxerrout


def aggregate_sntypes(reverse=False):
    if reverse:
        aggregate_map = {99: (45, 61, 62, 63, 90, 92),
                         1: (11, 1),
                         5: (3,13),
                         6: (2, 12, 14),
                         41: (41,),
                         43: (43,),
                         50: (50,),
                         51: (51,),
                         60: (60,),
                         64: (64,),
                         70: (70,),
                         80: (80,),
                         81: (81,),
                         83: (83,),
                         91: (91,),
                         93: (93,)}
    else:
        aggregate_map = {1: 1,
                         11: 1,
                         2: 6,
                         3: 5,
                         12: 6,
                         13: 5,
                         14: 6,
                         41: 41,
                         43: 43,
                         45: 45,
                         50: 50,
                         51: 51,
                         60: 60,
                         61: 61,
                         62: 62,
                         63: 63,
                         64: 64,
                         70: 70,
                         80: 80,
                         81: 81,
                         83: 83,
                         90: 90,
                         91: 91,
                         92: 92,
                         93: 93,
                         }

    return aggregate_map
