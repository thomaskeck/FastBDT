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

    # Train FastBDT using its PythonInterface, which is based on the SKLearn classifiers
    clf = FastBDT.Classifier(purityTransformation=1)
    clf.fit(X=X_train, y=y_train)
    p = clf.predict(X_test)
    global_auc = sklearn.metrics.roc_auc_score(y_test, p)
    print("Global AUC", global_auc)

    # Intern feature importance is calculated using the sum of the information gains
    # provided by each feature in all decision trees
    print("Intern Feature Importance")
    print(clf.internFeatureImportance())

    # Extern feature importance is calculated using the drop in the area under the receiver operating characteristics curve
    # if the most important feature is left out recursively
    print("Extern Feature Importance")
    print(clf.externFeatureImportance(X_train, y_train, None, X_test, y_test, None))

    # Individual feature importance is the sum of the information gains provided by feature
    # in the path an individual event takes through the forest
    print("Individual Feature Importance")
    events = [ np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
               np.array([2.0, 2.0, 3.0, 4.0, 5.0]),
               np.array([0.0, 2.0, 3.0, 4.0, 5.0]),
               np.array([1.0, 3.0, 3.0, 4.0, 5.0]),
               np.array([1.0, 1.0, 3.0, 4.0, 5.0]),
               np.array([1.0, 2.0, 4.0, 4.0, 5.0]),
               np.array([1.0, 2.0, 2.0, 4.0, 5.0]),
               np.array([1.0, 2.0, 3.0, 5.0, 5.0]),
               np.array([1.0, 2.0, 3.0, 3.0, 5.0]),
               np.array([1.0, 2.0, 3.0, 4.0, 6.0]),
               np.array([1.0, 2.0, 3.0, 4.0, 4.0]) ]

    for event in events:
        print(clf.individualFeatureImportance(event))
