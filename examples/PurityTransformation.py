from PyFastBDT import FastBDT

import pandas
import numpy as np
import sklearn.metrics

if __name__ == '__main__':

    data = np.arange(100000)
    X = (data % 100).reshape((100000, 1))
    y = (data % 2) == 1

    clf = FastBDT.Classifier(nTrees=1, depth=1, shrinkage=0.1, subsample=1.0, purityTransformation=[False]).fit(X=X, y=y)
    p = clf.predict(X)
    print('No Purity Transformation', sklearn.metrics.roc_auc_score(y, p))

    clf = FastBDT.Classifier(nTrees=1, depth=1, shrinkage=0.1, subsample=1.0, purityTransformation=[True]).fit(X=X, y=y)
    p = clf.predict(X)
    print('With Purity Transformation', sklearn.metrics.roc_auc_score(y, p))
