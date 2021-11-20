import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def normalize_z(df):
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    return (df - mean) / std


def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names]
    df_target = df[target_names]
    return df_feature, df_target


def compute_cost(X, y, beta):
    m = len(y)
    # print('X:', X)
    # print('beta', beta)
    J = (1/(2*m))*(np.square(np.matmul(X, beta) - y)).sum(axis=0)
    return J


def prepare_feature(df_feature):
    m, n = df_feature.shape
    feature_np = df_feature.to_numpy()
    feature_np = feature_np.reshape(m,n)
    feature_np = np.concatenate((np.ones((m,1)), feature_np), axis=1)
    return feature_np


def prepare_target(df_target):
    target_np = df_target.to_numpy()
    return target_np


def gradient_descent(X, y, beta, alpha, num_iters):
    m = len(y)
    J_storage = []
    for iter in range(num_iters):
        J_storage.append(compute_cost(X, y, beta))
        hx = np.matmul(X, beta)
        beta = beta - alpha*(1/m)*(np.matmul(X.T, hx - y))
    return beta, J_storage


def predict_norm(X, beta):
    return np.matmul(X, beta)


def predict(df_feature, beta):
    df_feature = normalize_z(df_feature)
    df_feature = prepare_feature(df_feature)
    return predict_norm(df_feature, beta)


def r2_score(y, ypred):
    ssres = np.square(y - ypred).sum(axis=0)
    sstot = np.square(y - np.mean(y)).sum(axis=0)
    return 1 - ssres / sstot


def mean_squared_error(target, pred):
    n = len(target)
    mse = (1/n)*(np.square(target-pred)).sum(axis=0)
    return mse


def split_data(df_feature, df_target, random_state=None, test_size=0.5):
    index = df_feature.index
    test_indices = np.random.RandomState(random_state).choice(index, size=int((len(index)) * test_size), replace=False)

    df_feature_train = df_feature[~df_feature.index.isin(test_indices)]
    df_target_train = df_target[~df_target.index.isin(test_indices)]

    df_feature_test = df_feature.loc[test_indices, :]
    df_target_test = df_target.loc[test_indices, :]

    return df_feature_train, df_feature_test, df_target_train, df_target_test