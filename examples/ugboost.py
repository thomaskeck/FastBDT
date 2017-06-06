import sys
from PyFastBDT import FastBDT
from PyFastBDT import utility

import numpy as np
import numpy
import numpy.linalg
import sklearn.metrics

import matplotlib.pyplot as plt
import matplotlib as mpl


def calculate_cdf_and_pdf(X):
    """
    Calculates cdf and pdf of given sample and adds under/overflow bins
        @param X 1-d numpy.array
    """
    pdf, bins = numpy.histogram(X, bins=100, density=True)
    cdf = numpy.cumsum(pdf * (bins - numpy.roll(bins, 1))[1:])
    return numpy.hstack([0.0, cdf, 1.0]), numpy.hstack([0.0, pdf, 0.0]), bins


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


def combine_probabilities(p1, p2):
    return p1*p2 / (p1*p2 + (1-p1)*(1-p2))


def evaluation(label, X_test, y_test, p, p_prior):
    print(label, utility.auc_roc(p, y_test), utility.flatness(p, X_test[:, 0], y_test, classes=[0]), utility.flatness(p, X_test[:, 0], y_test, classes=[1]))
    print(label, sklearn.metrics.roc_auc_score(y_test, p))
    print(label + " with prior", sklearn.metrics.roc_auc_score(y_test, combine_probabilities(p, p_prior)))
    plt.scatter(X_test[y_test == 1, 0], p[y_test == 1], c='r', label=label + " (Signal)", alpha=0.2)
    plt.scatter(X_test[y_test == 0, 0], p[y_test == 0], c='b', label=label + " (Background)", alpha=0.2)
    plt.xlabel("Feature")
    plt.ylabel("Probability")
    plt.show()


if __name__ == '__main__':
    # Create some Monte Carlo data using a multidimensional gaussian distribution
    # The 0th row of the coveriance matrix describes the correlation to the target variable
    mean = [0.5, 0.4, 0.4]
    cov = [[1.0, 0.6, 0.6],
           [0.0, 1.0, 0.0],
           [0.0, 0.0, 1.0]]
    
    mean2 = [-0.5, -0.4, -0.4]
    cov2 = [[1.0, 0.6, 0.6],
           [0.0, 1.0, 0.0],
           [0.0, 0.0, 1.0]]

    for i in range(len(mean)):
        for j in range(i+1, len(mean)):
            cov[j][i] = cov[i][j]
            cov2[j][i] = cov2[i][j]

    N_train, N_test = 100000, 2000
    data = np.random.multivariate_normal(mean2, cov2, (N_train + N_test)//2)
    data2 = np.random.multivariate_normal(mean, cov, (N_train + N_test)//2)
    X_train, y_train = np.r_[data[:N_train//2], data2[:N_train//2]], np.r_[np.ones(N_train//2) == 1, np.ones(N_train//2) == 0]
    X_test, y_test = np.r_[data[N_train//2:], data2[N_train//2:]], np.r_[np.ones(N_test//2) == 1, np.ones(N_test//2) == 0]

    # First variable is the variable we want to have independent of our network output
    prior = Prior(X_train[y_train == 1, 0], X_train[y_train == 0, 0])
    p_prior = prior.get_prior(X_test[:, 0])
    evaluation("Prior", X_test, y_test, p_prior, p_prior)
    
    evaluation("Random", X_test, y_test, np.random.uniform(size=N_test), p_prior)

    for i in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
        p = FastBDT.Classifier(flatnessLoss=i, numberOfFlatnessFeatures=1).fit(X=np.c_[X_train[:, 1:], X_train[:, 0]], y=y_train).predict(X_test[:, 1:])
        print("Flatness", i)
        evaluation("UBoost", X_test, y_test, p, p_prior)
    
    p = FastBDT.Classifier().fit(X=X_train, y=y_train).predict(X_test)
    evaluation("Full", X_test, y_test, p, p_prior)

    p = FastBDT.Classifier().fit(X=X_train[:, 1:], y=y_train).predict(X_test[:, 1:])
    evaluation("Restricted", X_test, y_test, p, p_prior)
        
