import sys
sys.path.append('../python/')
import FastBDT

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
        return self.signal_pdf[numpy.digitize(X, bins=self.signal_bins)]

    def get_bckgrd_pdf(self, X):
        return self.bckgrd_pdf[numpy.digitize(X, bins=self.bckgrd_bins)]

    def get_signal_cdf(self, X):
        return self.signal_cdf[numpy.digitize(X, bins=self.signal_bins)]

    def get_bckgrd_cdf(self, X):
        return self.bckgrd_cdf[numpy.digitize(X, bins=self.bckgrd_bins)]
    
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
    #plt.plot(train_fpr, train_tpr, label=label + ' (Train) ROC Integral = {:.3}'.format(train_auc))
    test_fpr, test_tpr, test_thresholds = sklearn.metrics.roc_curve(test_truth, test_prediction)
    test_auc = sklearn.metrics.auc(test_fpr, test_tpr)
    plt.plot(test_fpr, test_tpr, lw=4, label=label + '  ROC Integral = {:.3}'.format(test_auc))
    #plt.legend()
    #plt.show()
    return train_auc, test_auc


def combine_probabilities(p1, p2):
    return p1*p2 / (p1*p2 + (1-p1)*(1-p2))


if __name__ == '__main__':
    train_datafile = '../files/D0Test.txt'
    data = pandas.DataFrame.from_csv(train_datafile, sep=' ', index_col=None)
    df = data[data['distance'] < 0.1]
    N = len(df) / 2
    train_df = df.iloc[:N]
    print("Length training data", len(train_df))
    test_df = df.iloc[N:]
    print("Length test data", len(test_df))
    #keys = ['dM', 'chiProb', 'distance', 'gamma1E', 'gamma2E', 'gamma1clusterTiming', 'gamma2clusterTiming', 'gamma1E9E25', 'gamma2E9E25', 'nTracks']
    keys = ['dM', 'Kpi0M', 'KpiM', 'chiProb', 'distance', 'gamma1E', 'gamma2E', 'gamma1clusterTiming', 'gamma2clusterTiming', 'gamma1E9E25', 'gamma2E9E25', 'nTracks']
    #keys = ['dM', 'Kpi0M', 'KpiM', 'chiProb', 'distance', 'gamma1E', 'gamma2E', 'gamma1clusterTiming', 'gamma2clusterTiming', 'gamma1E9E25', 'gamma2E9E25', 'nTracks', 'dMBestCandidate']

    signal = train_df[train_df.isSignal == 1][keys[0]].values
    bckgrd = train_df[train_df.isSignal == 0][keys[0]].values
    prior = Prior(signal, bckgrd)
    splot_weights = calculate_splot_weights([prior.get_signal_pdf(train_df[keys[0]].values), prior.get_bckgrd_pdf(train_df[keys[0]].values)], [len(signal), len(bckgrd)])

    full_forest = FastBDT.Classifier().fit(X=train_df[keys].values,
                                           y=train_df['isSignal'].values)

    ordinary_forest = FastBDT.Classifier().fit(X=train_df[keys[1:]].values,
                                               y=train_df['isSignal'].values)

    splot_forest = FastBDT.Classifier().fit(X=numpy.r_[train_df[keys[1:]].values, train_df[keys[1:]].values],
                                             y=numpy.r_[numpy.ones(N), numpy.zeros(N)],
                                             weights=prior.get_splot_weights(train_df[keys[0]].values))

    boost_forest = FastBDT.Classifier().fit(X=numpy.r_[train_df[keys[1:]].values, train_df[keys[1:]].values],
                                            y=numpy.r_[numpy.ones(N), numpy.zeros(N)],
                                            weights=prior.get_boost_weights(train_df[keys[0]].values))
    aplot_forest = FastBDT.Classifier().fit(X=numpy.r_[train_df[keys[1:]].values, train_df[keys[1:]].values],
                                            y=numpy.r_[numpy.ones(N), numpy.zeros(N)],
                                            weights=prior.get_aplot_weights(train_df[keys[0]].values, boost_forest.predict(train_df[keys[1:]].values))) 
 

    # Side-Band Subtraction
    signal_region = (train_df.dM.abs() < 0.05)
    neg_signal_region = (0.24 < train_df.dM.abs()) & (train_df.dM.abs() < 0.2849)
    bckgrd_region = (0.258 < train_df.dM.abs()) & (train_df.dM.abs() < 0.2746)
    print("SignalRegion:", "Signal", (signal_region & (train_df.isSignal == 1)).sum(), "Bckgrd", (signal_region & (train_df.isSignal == 0)).sum())
    print("BckgrdRegion:", "Signal", (bckgrd_region & (train_df.isSignal == 1)).sum(), "Bckgrd", (bckgrd_region & (train_df.isSignal == 0)).sum())
    print("NegSignalRegion:", "Signal", (neg_signal_region & (train_df.isSignal == 1)).sum(), "Bckgrd", (neg_signal_region & (train_df.isSignal == 0)).sum())

    side_forest = FastBDT.Classifier().fit(X=numpy.r_[train_df[signal_region][keys[1:]].values, train_df[bckgrd_region][keys[1:]].values, train_df[neg_signal_region][keys[1:]].values],
                                            y=numpy.r_[numpy.ones(signal_region.sum()), numpy.zeros(bckgrd_region.sum()), numpy.ones(neg_signal_region.sum())],
                                            weights=numpy.r_[numpy.ones(signal_region.sum()), numpy.ones(bckgrd_region.sum()), -numpy.ones(neg_signal_region.sum())]) 


    seaborn.set(font_scale=4.5)
    seaborn.distplot(train_df.dM.values, bins=200, kde=False, hist_kws={'range': (-0.3, 0.3)}, label='Data')
    seaborn.distplot(train_df[signal_region].dM.values, bins=200, kde=False, hist_kws={'range': (-0.3, 0.3)}, label='Signal Region')
    seaborn.distplot(train_df[neg_signal_region].dM.values, bins=200, kde=False, hist_kws={'range': (-0.3, 0.3)}, label='Negative Signal Region')
    seaborn.distplot(train_df[bckgrd_region].dM.values, bins=200, kde=False, hist_kws={'range': (-0.3, 0.3)}, label='Background Region')
    plt.xlim((-0.3,0.3))
    plt.xlabel('Reconstructed Mass - Nominal Mass')
    plt.legend()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(24, 16)
    plt.savefig('sideband.png')
    plt.clf()
    
    seaborn.distplot(train_df.dM.values, bins=200, kde=False, hist_kws={'range': (-0.3, 0.3)}, label='Signal Fit')
    seaborn.distplot(train_df[train_df.isSignal == 0].dM.values, kde=False, bins=200, hist_kws={'range': (-0.3, 0.3)}, label='Background Fit')
    plt.xlim((-0.3,0.3))
    plt.xlabel('Reconstructed Mass - Nominal Mass')
    plt.legend()
    figure = plt.gcf() # get current figure
    figure.set_size_inches(24, 16)
    plt.savefig('splot.png')
    plt.clf()

    full_prediction_train = full_forest.predict(train_df[keys].values)
    ordinary_prediction_train = ordinary_forest.predict(train_df[keys[1:]].values)
    splot_prediction_train = splot_forest.predict(train_df[keys[1:]].values)
    aplot_prediction_train = aplot_forest.predict(train_df[keys[1:]].values)
    prior_prediction_train = prior.get_prior(train_df[keys[0]].values)
    side_prediction_train = side_forest.predict(train_df[keys[1:]].values)
    ordinary_prior_prediction_train = combine_probabilities(ordinary_prediction_train, prior_prediction_train)
    splot_prior_prediction_train = combine_probabilities(splot_prediction_train, prior_prediction_train)
    aplot_prior_prediction_train = combine_probabilities(aplot_prediction_train, prior_prediction_train)
    side_prior_prediction_train = combine_probabilities(side_prediction_train, prior_prediction_train)
    truth_train = train_df['isSignal'].values
    
    full_prediction_test = full_forest.predict(test_df[keys].values)
    ordinary_prediction_test = ordinary_forest.predict(test_df[keys[1:]].values)
    splot_prediction_test = splot_forest.predict(test_df[keys[1:]].values)
    aplot_prediction_test = aplot_forest.predict(test_df[keys[1:]].values)
    prior_prediction_test = prior.get_prior(test_df[keys[0]].values)
    side_prediction_test = side_forest.predict(test_df[keys[1:]].values)
    ordinary_prior_prediction_test = combine_probabilities(ordinary_prediction_test, prior_prediction_test)
    splot_prior_prediction_test = combine_probabilities(splot_prediction_test, prior_prediction_test)
    aplot_prior_prediction_test = combine_probabilities(aplot_prediction_test, prior_prediction_test)
    side_prior_prediction_test = combine_probabilities(side_prediction_test, prior_prediction_test)
    truth_test = test_df['isSignal'].values

    seaborn.set_palette("Set1", n_colors=10, desat=.5)
    trivial_prior = train_df.isSignal.mean()
    #calculate_score("Trivial", numpy.ones(len(truth_train))*trivial_prior, numpy.ones(len(truth_test))*trivial_prior, truth_train, truth_test)
    calculate_score("Full", full_prediction_train, full_prediction_test, truth_train, truth_test)
    calculate_score("Ordinary", ordinary_prediction_train, ordinary_prediction_test, truth_train, truth_test)
    calculate_score("SPlot", splot_prediction_train, splot_prediction_test, truth_train, truth_test)
    calculate_score("APlot", aplot_prediction_train, aplot_prediction_test, truth_train, truth_test)
    calculate_score("Sideband", side_prediction_train, side_prediction_test, truth_train, truth_test)
    calculate_score("Prior", prior_prediction_train, prior_prediction_test, truth_train, truth_test)
    calculate_score("OrdinaryPrior", ordinary_prior_prediction_train, ordinary_prior_prediction_test, truth_train, truth_test)
    calculate_score("SPlotPrior", splot_prior_prediction_train, splot_prior_prediction_test, truth_train, truth_test)
    calculate_score("APlotPrior", aplot_prior_prediction_train, aplot_prior_prediction_test, truth_train, truth_test)
    calculate_score("SidePrior", side_prior_prediction_train, side_prior_prediction_test, truth_train, truth_test)
    plt.xlabel('False Positive Rate (Type I Error)')
    plt.ylabel('True Positive Rate (Efficiency)')
    plt.xlim((0.5,1.0))
    plt.xlim((0.0,0.5))
    plt.legend(loc='lower right')
    figure = plt.gcf() # get current figure
    figure.set_size_inches(24, 16)
    plt.savefig('splot_sideband_roc.png')
    plt.clf()
