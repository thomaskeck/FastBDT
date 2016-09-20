import sys
sys.path.append('python/')
import FastBDT

import numpy as np
import sklearn.metrics
#import matplotlib.pyplot as plt
#import matplotlib as mpl


def create_toy_events(nevents):
    events = []
    y = []
    for nevent in range(nevents):
        event = []
        if np.random.random() < 0.6:
            event.append(1.0)
            event.append(np.random.random())
            event.append(np.random.random())
            y.append(event[-1] < np.random.random())
        else:
            event.append(0.0)
            event.append(np.random.random())
            event.append(np.random.random())
            y.append(event[-2] < np.random.random())
        events.append(event)
    return np.array(events), np.array(y)


if __name__ == '__main__':

    N_train, N_test = 10000, 10000
    X_train, y_train = create_toy_events(N_train)
    X_test, y_test = create_toy_events(N_test)

    clf = FastBDT.Classifier().fit(X=X_train, y=y_train)
    p = clf.predict(X_test)
    global_auc = sklearn.metrics.roc_auc_score(y_test, p)
    print("Global AUC", global_auc)

    print("Intern Feature Importance")
    print(clf.internFeatureImportance())
    print("Extern Feature Importance")
    print(clf.externFeatureImportance(X_train, y_train, None, X_test, y_test, None))

    events = [ np.array([0.0, 0.2, 0.2]),
               np.array([1.0, 0.2, 0.2]) ]

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
