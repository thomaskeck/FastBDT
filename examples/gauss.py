import sys
sys.path.append('python/')
import FastBDT

import numpy as np
import sklearn.metrics
#import matplotlib.pyplot as plt
#import matplotlib as mpl

if __name__ == '__main__':

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

    clf = FastBDT.Classifier().fit(X=X_train, y=y_train)
    p = clf.predict(X_test)
    global_auc = sklearn.metrics.roc_auc_score(y_test, p)
    print("Global AUC", global_auc)

    print("Intern Feature Importance")
    print(clf.internFeatureImportance())
    print("Extern Feature Importance")
    print(clf.externFeatureImportance(X_train, y_train, None, X_test, y_test, None))

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

    print("Individual Feature Importance")
    for event in events:
        print(clf.individualFeatureImportance(event))
    #plt.plot(fpr, tpr, lw=4, label='ROC Integral = {:.3}'.format(auc))
    #plt.xlabel('False Positive Rate (Type I Error)')
    #plt.ylabel('True Positive Rate (Efficiency)')
    #plt.xlim((0.0,1.0))
    #plt.xlim((0.0,1.0))
    #plt.legend(loc='lower right')
    #plt.show()
    #figure = plt.gcf() # get current figure
    #figure.set_size_inches(24, 16)
