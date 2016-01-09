import sys
sys.path.append('../python/')

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy
import numpy.linalg
import pandas
import seaborn
import sklearn.metrics


class Prior(object):
    def __init__(self, signal, bckgrd):
        self.signal_cdf, self.signal_pdf, self.signal_bins = calculate_cdf_and_pdf(signal)
        self.bckgrd_cdf, self.bckgrd_pdf, self.bckgrd_bins = calculate_cdf_and_pdf(bckgrd)
        # Avoid numerical instabilities
        self.bckgrd_pdf[0] = self.bckgrd_pdf[-1] = 1
        self.signal_yield = len(signal)
        self.bckgrd_yield = len(bckgrd)

    def get_signal_pdf(self, X):
        return self.signal_pdf[numpy.digitize(X, bins=signal_bins)]

    def get_bckgrd_pdf(self, X):
        return self.bckgrd_pdf[numpy.digitize(X, bins=bckgrd_bins)]

    def get_signal_cdf(self, X):
        return self.signal_cdf[numpy.digitize(X, bins=signal_bins)]

    def get_bckgrd_cdf(self, X):
        return self.bckgrd_cdf[numpy.digitize(X, bins=bckgrd_bins)]
    
    def get_prior(self, X):
        return self.get_signal_pdf(X) / (self.get_signal_pdf(X) + self.get_bckgrd_pdf(X))
    
    def get_signal_boost_weights(self, X):
        return self.get_signal_cdf(X) / self.get_bckgrd_pdf(X)
    
    def get_bckgrd_boost_weights(self, X):
        # NOT self.get_bckgrd_cdf() here, signal and background are handlet asymmetrical!
        return (1.0 - self.get_signal_cdf(X)) / self.get_bckgrd_pdf(X)

    def get_boost_weights(self, X):
        return numpy.r_[self.get_signal_boost_weights(X), self.get_bckgrd_boost_weights(X)]

    def get_splot_weights(self, X):
        pdfs = [self.get_signal_pdf(X), self.get_bckgrd_pdf(X)]
        yields = [self.signal_yield, self.bckgrd_yield]
        weights = calculate_splot_weights(pdfs, yields)
        return numpy.r_[weights[0], weights[1]]
    
    def get_aplot_weights(self, X, boost_prediction):
        reg_boost_prediction = boost_prediction * 0.99 + 0.005
        weights = (self.get_signal_cdf(X) / reg_boost_prediction +  (1.0 - self.get_signal_cdf(X)) / (1.0 - reg_boost_prediction)) / 2
        return self.get_splot_weights(X) * numpy.r_[weights, weights]


def calculate_cdf_and_pdf(X):
    """
    Calculates cdf and pdf of given sample and adds under/overflow bins
        @param X 1-d numpy.array
    """
    pdf, bins = numpy.histogram(X, bins=100, density=True)
    cdf = numpy.cumsum(pdf * (bins - numpy.roll(bins, 1))[1:])
    return numpy.hstack([0.0, cdf, 1.0]), numpy.hstack([0.0, pdf, 0.0]), bins


def calculate_splot_weights(pdfs, yields):
    """
    Calculates sPlot weights using the pdfs
        @param pdfs list of 1-d numpy.array with pdf values of the different components for each event
        @param yields list of the yields of the different components
    """
    N_components = len(pdfs)
    # Consistency checks
    if N_components != len(yields):
        raise RuntimeError("You have to provide the same number of pdfs and yields!")
    if N_components < 2:
        raise RuntimeError("Need at least two components!")

    # Calculate covariance matrix
    inverse_covariance = numpy.zeros((N_components, N_components))
    norm = sum((yields[k] * pdfs[k] for k in range(1, N_components)), yields[0] * pdfs[0])**2
    for i in range(N_components):
        for j in range(N_components):
            inverse_covariance[i, j] = numpy.nansum(pdfs[i] * pdfs[j] / norm)
    covariance = numpy.linalg.inv(inverse_covariance)

    # Return list of sPlot weights for each component
    return [sum(covariance[n, k] * pdfs[k] for k in range(N_components)) /
            sum(yields[k] * pdfs[k] for k in range(N_components)) for n in range(N_components)]


def calculate_score(label, train_prediction, test_prediction, train_truth, test_truth):
    train_fpr, train_tpr, train_thresholds = sklearn.metrics.roc_curve(train_truth, train_prediction)
    train_auc = sklearn.metrics.auc(train_fpr, train_tpr)
    plt.plot(train_fpr, train_tpr, label=label + ' (Train) ROC Integral = {:.3}'.format(train_auc))
    test_fpr, test_tpr, test_thresholds = sklearn.metrics.roc_curve(test_truth, test_prediction)
    test_auc = sklearn.metrics.auc(test_fpr, test_tpr)
    plt.plot(test_fpr, test_tpr, label=label + '  (Test) ROC Integral = {:.3}'.format(test_auc))
    plt.legend()
    plt.show()
    return auc


def combine_probabilities(p1, p2):
    return p1*p2 / (p1*p2 + (1-p1)*(1-p2))


if __name__ == '__main__':
    train_datafile = 'small.txt'
    data = pandas.DataFrame.from_csv(train_datafile, sep=' ', index_col=None)
    df = data[data['distance'] < 0.1]
    N = 4500
    train_df = df.iloc[:N]
    test_df = df.iloc[N:]
    keys = ['dM', 'Kpi0M', 'KpiM', 'chiProb', 'distance', 'gamma1E', 'gamma2E', 'gamma1clusterTiming', 'gamma2clusterTiming', 'gamma1E9E25', 'gamma2E9E25', 'nTracks']
    #keys = ['dM', 'Kpi0M', 'KpiM', 'chiProb', 'distance', 'gamma1E', 'gamma2E', 'gamma1clusterTiming', 'gamma2clusterTiming', 'gamma1E9E25', 'gamma2E9E25', 'nTracks', 'dMBestCandidate']

    signal = train_df[train_df.isSignal == 1].dM.values
    bckgrd = train_df[train_df.isSignal == 0].dM.values
    prior = Prior(signal, bckgrd)
    splot_weights = calculate_splot_weights([prior.get_signal_pdf(train_df.dM.values), prior.get_bckgrd_pdf(train_df.dM.values)], [len(signal), len(bckgrd)])

    full_forest = FastBDT.Classifier().fit(X=train_df[keys].values,
                                               y=train_df['isSignal'].values)

    ordinary_forest = FastBDT.Classifier().fit(X=train_df[keys[1:]].values,
                                               y=train_df['isSignal'].values)

    splot_forest = FastBDT.Classifier().fit(X=numpy.r_[train_df[keys[1:]].values, train_df[keys[1:]].values],
                                             y=numpy.r_[numpy.ones(N), numpy.zeros(N)],
                                             weights=prior.get_splot_weights(train_df.dM.values))

    boost_forest = FastBDT.Classifier().fit(X=numpy.r_[train_df[keys[1:]].values, train_df[keys[1:]].values],
                                            y=numpy.r_[numpy.ones(N), numpy.zeros(N)],
                                            weights=prior.get_boost_weights(train_df.dM.values))
    aplot_forest = FastBDT.Classifier().fit(X=numpy.r_[train_df[keys[1:]].values, train_df[keys[1:]].values],
                                            y=numpy.r_[numpy.ones(N), numpy.zeros(N)],
                                            weights=prior.get_aplot_weights(train_df.dM.values, boost_forest.predict(train_df[keys[1:]].values))) 
  

    full_prediction_train = full_forest.predict(train_df[keys].values)
    ordinary_prediction_train = ordinary_forest.predict(train_df[keys[1:]].values)
    splot_prediction_train = splot_forest.predict(train_df[keys[1:]].values)
    aplot_prediction_train = aplot_forest.predict(train_df[keys[1:]].values)
    prior_prediction_train = prior.get_prior(train_df.dM.values)
    ordinary_prior_prediction_train = combine_probabilities(ordinary_prediction_train, prior_prediction_train)
    splot_prior_prediction_train = combine_probabilities(splot_prediction_train, prior_prediction_train)
    aplot_prior_prediction_train = combine_probabilities(aplot_prediction_train, prior_prediction_train)
    truth_train = train_df['isSignal'].values
    
    full_prediction_test = full_forest.predict(test_df[keys].values)
    ordinary_prediction_test = ordinary_forest.predict(test_df[keys[1:]].values)
    splot_prediction_test = splot_forest.predict(test_df[keys[1:]].values)
    aplot_prediction_test = aplot_forest.predict(test_df[keys[1:]].values)
    prior_prediction_test = prior.get_prior(test_df.dM.values)
    ordinary_prior_prediction_test = combine_probabilities(ordinary_prediction_test, prior_prediction_test)
    splot_prior_prediction_test = combine_probabilities(splot_prediction_test, prior_prediction_test)
    aplot_prior_prediction_test = combine_probabilities(aplot_prediction_test, prior_prediction_test)
    truth_test = test_df['isSignal'].values

    trivial_prior = train_df.isSignal.mean()
    calculate_score("Trivial", numyp.ones(len(truth_train))*trivial_prior, numyp.ones(len(truth_test))*trivial_prior, truth_train, truth_test)
    calculate_score("Full", full_prediction_train, full_prediction_test, truth_train, truth_test)
    calculate_score("Ordinary", ordinary_prediction_train, ordinary_prediction_test, truth_train, truth_test)
    calculate_score("SPlot", splot_prediction_train, splot_prediction_test, truth_train, truth_test)
    calculate_score("APlot", aplot_prediction_train, aplot_prediction_test, truth_train, truth_test)
    calculate_score("Prior", prior_prediction_train, prior_prediction_test, truth_train, truth_test)
    calculate_score("OrdinaryPrior", ordinary_prior_prediction_train, ordinary_prior_prediction_test, truth_train, truth_test)
    calculate_score("SPlotPrior", splot_prior_prediction_train, splot_prior_prediction_test, truth_train, truth_test)
    calculate_score("APlotPrior", aplot_prior_prediction_train, aplot_prior_prediction_test, truth_train, truth_test)
