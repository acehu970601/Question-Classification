from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import fbeta_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import imblearn as ib

def split_data(features, fine_label, coarse_label, label_type, test_size = 0.15, val_size = 0.2):
    '''
    split data into training / validation / test
    '''
    if label_type == 'coarse':
        label = coarse_label
    else:
        label = fine_label

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = test_size, random_state=1)

    X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size = val_size, random_state=1) # 0.25 x 0.8 = 0.2

    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes = np.linspace(.1, 1.0, 10)):
    """
    Generate a simple plot of the test and traning learning curve.
    """

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True)
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()

def plot_validation_curve(estimator, model_name, X, y, param_name, param_range = np.linspace(1.0, 10.0, 5)):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        scoring=aucroc_scorer)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with " + model_name)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

def aucroc_scorer(estimator, X, y):
    '''
    Calculate AUCROC scores
    '''
    prob = estimator.predict_proba(X)
    prob[prob < 0] = 0

    y_binarized = label_binarize(y, classes = range(len(set(y))))
    aucroc_scores = roc_auc_score(y_binarized, prob, multi_class = 'ovr')
    return aucroc_scores

def data_augmentation(X, y):
    '''
    Data augmentation by SMOTE
    '''
    oversample = ib.over_sampling.SMOTE(sampling_strategy = 'not majority', k_neighbors = 2)
    X, y = oversample.fit_resample(X,y)
    return X,y
