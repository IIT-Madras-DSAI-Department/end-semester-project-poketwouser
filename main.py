import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from algorithms import *

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')

    targetcol1 = 'label'
    targetcol2 = 'even'

    Xtrain = dftrain[featurecols]
    ytrain = dftrain[targetcol1]
    ytrain2 = dftrain[targetcol2]
    
    Xval = dfval[featurecols]
    yval = dfval[targetcol1]
    yval2 = dfval[targetcol2]

    return (Xtrain, Xval, ytrain, yval)

Xtrain, Xval, ytrain, yval = read_data()

pca = PCA(n_components=30)
Xtrain_pca = pca.fit_transform(Xtrain)
Xval_pca = pca.transform(Xval)

pca_xgb = PCA(n_components=28)
Xtrain_xgb = pca_xgb.fit_transform(Xtrain)
Xval_xgb = pca_xgb.transform(Xval)

def train_ovr_knn(X_pca, y):
    ovr_knn = {}
    for d in range(10):
        y_bin = (y == d).astype(int)
        knn = KNN(k=1)
        knn.fit(X_pca, y_bin)
        ovr_knn[d] = knn
    return ovr_knn


def train_evenodd_knn(X_pca, y):
    y_evenodd = (y % 2).astype(int)
    knn = KNN(k=1)
    knn.fit(X_pca, y_evenodd)
    return knn


def train_parity_ovo_knn(X_pca, y, group):
    pair_knn = {}
    group = sorted(group)
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            a, b = group[i], group[j]

            mask = (y == a) | (y == b)
            X_pair = X_pca[mask]
            y_pair = y[mask]

            knn = KNN(k=1)
            knn.fit(X_pair, y_pair)
            pair_knn[(a, b)] = knn

    return pair_knn


def predict_ovr(ovr_knn, X_pca):
    n = len(X_pca)
    y_pred = np.zeros(n, dtype=int)

    for i in range(n):
        scores = {d: ovr_knn[d].predict(X_pca[i:i+1])[0] for d in range(10)}
        y_pred[i] = max(scores, key=scores.get)

    return y_pred


def predict_evenodd(knn_evenodd, X_pca):
    return knn_evenodd.predict(X_pca)


def refine_using_parity_ovo(X_single, predicted_digit, ovo_models):
    votes = []
    for (a, b), knn in ovo_models.items():
        vote = knn.predict(X_single)[0]
        votes.append(vote)
    return max(set(votes), key=votes.count)


def full_pipeline_predict(X_pca, y_pred_ovr, y_pred_evenodd, ovo_even, ovo_odd):
    final_pred = y_pred_ovr.copy()

    even_digits = {0,2,4,6,8}
    odd_digits  = {1,3,5,7,9}

    for i in range(len(X_pca)):
        d = y_pred_ovr[i]
        parity_expected = d % 2
        parity_pred = y_pred_evenodd[i]

        if parity_expected == parity_pred:
            continue

        if parity_pred == 0:
            final_pred[i] = refine_using_parity_ovo(X_pca[i:i+1], d, ovo_even)
        else:
            final_pred[i] = refine_using_parity_ovo(X_pca[i:i+1], d, ovo_odd)

    return final_pred

def train_full_parity_pipeline(Xtrain, ytrain, pca):

    Xtrain_pca = pca.fit_transform(Xtrain)

    ovr_knn = train_ovr_knn(Xtrain_pca, ytrain)

    knn_evenodd = train_evenodd_knn(Xtrain_pca, ytrain)

    pca_xgb = PCA(n_components=28)
    Xtrain_xgb = pca_xgb.fit_transform(Xtrain)
    y_evenodd_train = (ytrain % 2).astype(int)

    xgb = XGBoostClassifier(
        n_estimators=105,
        learning_rate=0.3,
        max_depth=9,
        lambda_l2=0.1,
        gamma=0,
        min_child_weight=1
    )
    xgb.fit(Xtrain_xgb, y_evenodd_train)

    even_group = {0,2,4,6,8}
    odd_group  = {1,3,5,7,9}

    ovo_even = train_parity_ovo_knn(Xtrain_pca, ytrain, even_group)
    ovo_odd  = train_parity_ovo_knn(Xtrain_pca, ytrain, odd_group)

    return ovr_knn, knn_evenodd, ovo_even, ovo_odd, pca, pca_xgb, xgb, y_evenodd_train

def run_full_parity_pipeline(Xtrain, ytrain, Xval, yval):
    pca = PCA(n_components=30)
    models = train_full_parity_pipeline(Xtrain, ytrain, pca)
    (ovr_knn, knn_evenodd, ovo_even, ovo_odd, pca, pca_xgb, xgb, y_evenodd_train) = models

    Xval_pca = pca.transform(Xval)
    Xval_xgb = pca_xgb.transform(Xval)
    y_evenodd_xgb = xgb.predict(Xval_xgb)

    y_evenodd_val = (yval % 2).astype(int)
    agreement = np.mean(y_evenodd_xgb == y_evenodd_val)

    y_ovr = predict_ovr(ovr_knn, Xval_pca)
    y_evenodd_knn = predict_evenodd(knn_evenodd, Xval_pca)

    final = full_pipeline_predict(Xval_pca, y_ovr, y_evenodd_knn, ovo_even, ovo_odd)

    return final, y_ovr, y_evenodd_knn, y_evenodd_xgb, models

def read_test_data(testfile='MNIST_test.csv'):
    
    dftest = pd.read_csv(testfile)

    featurecols = list(dftest.columns)
    featurecols.remove('label')
    featurecols.remove('even')

    targetcol = 'label'

    Xtest = dftest[featurecols]
    ytest = dftest[targetcol]

    return (Xtest, ytest)

Xtest, ytest = read_test_data('MNIST_test.csv')

final, y_ovr, y_evenodd_knn, y_evenodd_xgb, models = run_full_parity_pipeline(Xtrain, ytrain, Xtest, ytest)

print(f1_score(ytest, final, average='weighted'))