import sys
sys.path.append('python/')
import FastBDT

import numpy as np
import sklearn.metrics
#import matplotlib.pyplot as plt
#import matplotlib as mpl

if __name__ == '__main__':

    mean = [0.0, 1.0, 2.0, 3.0]
    cov = [[1.0, 0.6, 0.4, 0.2],
           [0.0, 1.0, 0.99, 0.0],
           [0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 1.0]]
    for i in range(len(mean)):
        for j in range(i+1, len(mean)):
            cov[j][i] = cov[i][j]
    N_train, N_test = 10000, 10000
    data = np.random.multivariate_normal(mean, cov, N_train + N_test)
    X_train, y_train = data[:N_train, 1:], data[:N_train, 0] > 0 
    X_test, y_test = data[N_train:, 1:], data[N_train:, 0] > 0 

    clf = FastBDT.Classifier().fit(X=X_train, y=y_train)
    p = clf.predict(X_test)
    print(clf.variableRanking())
    global_auc = sklearn.metrics.roc_auc_score(y_test, p)
    print("Global AUC", global_auc)

    for i in range(3):
        print("Without feature", i)
        X_train_temp = np.delete(X_train, i, axis=1)
        X_test_temp = np.delete(X_test, i, axis=1)
        clf = FastBDT.Classifier().fit(X=X_train_temp, y=y_train)
        p = clf.predict(X_test_temp)
        print(clf.variableRanking())
        auc = sklearn.metrics.roc_auc_score(y_test, p)
        print("AUC", auc, (global_auc - auc)/global_auc)


    #plt.plot(fpr, tpr, lw=4, label='ROC Integral = {:.3}'.format(auc))
    #plt.xlabel('False Positive Rate (Type I Error)')
    #plt.ylabel('True Positive Rate (Efficiency)')
    #plt.xlim((0.0,1.0))
    #plt.xlim((0.0,1.0))
    #plt.legend(loc='lower right')
    #plt.show()
    #figure = plt.gcf() # get current figure
    #figure.set_size_inches(24, 16)
