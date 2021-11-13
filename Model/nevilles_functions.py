import random
import statistics
import pandas as pd
import numpy as np
import cohort_functions as cf


# Splits list into n equal parts
def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]


# k-folds cross validation
# https://medium.datadriveninvestor.com/k-fold-cross-validation-for-parameter-tuning-75b6cb3214f
def cross_val_score(df_feature, df_target, beta, alpha, num_iters, random_state=100, cv=10):
    index = df_feature.index
    index = list(index)

    # Shuffles the indexes
    np.random.RandomState(random_state).shuffle(index)

    # Split indexes into cv parts
    folds = partition(index, cv)
    r2_scores = []
    mse_scores = []

    # Iterates through cv parts, each time using each part as test set/rest as train set
    # records r2/mse value and adds to scores
    for fold in folds:
        print('fold:', fold)
        # basically train test split here
        df_feature_train = df_feature[~df_feature.index.isin(fold)]
        df_target_train = df_target[~df_target.index.isin(fold)]
        df_feature_test = df_feature.iloc[fold, :]
        df_target_test = df_target.iloc[fold, :]

        X = df_feature_train
        X = cf.prepare_feature(X)
        y = cf.prepare_target(df_target_train)

        beta, J_storage = cf.gradient_descent(X, y, beta, alpha, num_iters)
        print('Cost:', J_storage[0], J_storage[-1])
        pred = cf.predict(df_feature_test, beta)
        target = cf.prepare_target(df_target_test)
        mse = cf.mean_squared_error(target, pred).item(0)

        mse_scores.append(mse)
    return statistics.mean(mse_scores)
