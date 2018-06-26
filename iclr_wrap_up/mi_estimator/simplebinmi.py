# Simplified MI computation code from https://github.com/ravidziv/IDNNs
import numpy as np
from functools import reduce, partial

flatten = lambda l: [item for sublist in l for item in sublist]

def histedges_equalN(x, nbin):
    bins = np.interp(np.linspace(0, len(x), nbin + 1), np.arange(len(x)), np.sort(x))
    err = reduce(lambda a, b: -1 if ((a == b) or (a == -1)) else b, bins) == -1
    if err:
        print("I cannot do a good binning, one unique value appears too often!")
    return bins



def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse


def bin_calc_information2(labelixs, layerdata, binsize):
    # This is even further simplified, where we use np.floor instead of digitize
    def get_h(d):
        digitized = np.floor(d / binsize).astype('int')
        p_ts, _ = get_unique_probs(digitized)
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(layerdata)
    H_LAYER_GIVEN_OUTPUT = 0
    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(layerdata[ixs, :])
    return H_LAYER, H_LAYER - H_LAYER_GIVEN_OUTPUT


def bin_calc_information_evenbins(labelixs, layerdata, numbins):
    # This is even further simplified, where we use np.floor instead of digitize
    def get_h(d):
        def get_bin(x, bins):
            for num, bin in enumerate(bins):
                if x < bin:
                    return num
        digitize = partial(get_bin, bins=histedges_equalN(flatten(d), numbins))
        digitized = []
        for line in d:
            l2 = [digitize(cell) for cell in line]
            digitized.append(l2)
        digitized = np.array(digitized, dtype=np.float64)
        p_ts, _ = get_unique_probs(digitized)
        return -np.sum(p_ts * np.log(p_ts))

    H_LAYER = get_h(layerdata)
    H_LAYER_GIVEN_OUTPUT = 0
    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_h(layerdata[ixs, :])
    return H_LAYER, H_LAYER - H_LAYER_GIVEN_OUTPUT
