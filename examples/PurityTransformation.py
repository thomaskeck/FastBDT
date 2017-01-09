import sys
from PyFastBDT import FastBDT

import numpy as np
import sklearn.metrics

if __name__ == '__main__':

    # Create some Monte Carlo data using a multidimensional gaussian distribution
    # The 0th row of the coveriance matrix describes the correlation to the target variable
    mean = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    cov = [[1.0, 0.8, 0.4, 0.2, 0.1, 0.0],
           [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]

    for i in range(len(mean)):
        for j in range(i+1, len(mean)):
            cov[j][i] = cov[i][j]

    N_train, N_test = 10000, 10000
    data = np.random.multivariate_normal(mean, cov, N_train + N_test)
    X_train, y_train = data[:N_train, 1:], data[:N_train, 0] > 0 
    X_test, y_test = data[N_train:, 1:], data[N_train:, 0] > 0 

    for pt in [0,1,2]:
        clf = FastBDT.Classifier(purityTransformation=pt)
        clf.fit(X=X_train, y=y_train)
        p = clf.predict(X_test)
        print("Purity Transformation = {}".format(pt), sklearn.metrics.roc_auc_score(y_test, p))
        print(clf.externFeatureImportance(X_train, y_train, None, X_test, y_test, None))
