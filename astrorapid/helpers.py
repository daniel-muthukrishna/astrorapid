import numpy as np


def find_nearest(array, value):
    """
    Find the index nearest to a given value.
    Adapted from: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx


def delete_indexes(deleteindexes, *args):
    newarrs = []
    for arr in args:
        newarr = np.delete(arr, deleteindexes)
        newarrs.append(newarr)

    return newarrs


def convert_lists_to_arrays(*args):
    output = []
    for arg in args:
        out_array = np.asarray(arg)
        output.append(out_array)

    return output


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


def get_sntypes():
    sntypes_map = {1: 'SNIa-norm',
                   11: 'SNIa-norm',
                   2: 'SNII',
                   12: 'SNIIpca',
                   14: 'SNIIn',
                   3: 'SNIbc',
                   13: 'SNIbc',
                   5: 'SNIbc',
                   6: 'SNII',
                   41: 'SNIa-91bg',
                   43: 'SNIa-x',
                   45: 'point-Ia',
                   50: 'Kilonova-GW170817',
                   51: 'Kilonova',
                   60: 'SLSN-I',
                   61: 'PISN',
                   62: 'ILOT',
                   63: 'CART',
                   64: 'TDE',
                   70: 'AGN',
                   80: 'RRLyrae',
                   81: 'Mdwarf',
                   83: 'EBE',
                   84: 'Mira',
                   90: 'uLens-BSR',
                   91: 'uLens-1STAR',
                   92: 'uLens-String',
                   93: 'uLens - Point',
                   99: 'Rare'}
    return sntypes_map
