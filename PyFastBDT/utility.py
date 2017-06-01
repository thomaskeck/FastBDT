import numpy as np


def flatness(feature, probability, target):
    quantiles = list(range(101))
    flatness_score = 0
    for m in [target == 1, target == 0]: 
        binning_feature = np.unique(np.percentile(feature[m], q=quantiles))
        binning_probability = np.unique(np.percentile(probability[m], q=quantiles))

        hist_n, _ = np.histogramdd(np.c_[probability[m], feature[m]],
                                   bins=[binning_probability, binning_feature])
        hist_inc = hist_n.sum(axis=1)
        hist_n /= hist_n.sum(axis=0)
        hist_inc /= hist_inc.sum(axis=0)
        flatness_score += (hist_n.T - hist_inc).std()
    return flatness_score


def auc_roc(probability, target):
    N = len(target)
    T = np.sum(target)
    index = np.argsort(probability)
    efficiency = (T - np.cumsum(target[index])) / float(T)
    purity = (T - np.cumsum(target[index])) / (N - np.cumsum(np.ones(N)))
    purity = np.where(np.isnan(purity), 0, purity)
    return np.abs(np.trapz(purity, efficiency))
