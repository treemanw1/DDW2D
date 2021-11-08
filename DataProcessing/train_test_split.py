import random
import numpy as np
import pandas as pd


def split_data(df, features, random_state=None, test_size=0.5):
    df_label = pd.DataFrame(df['Deaths'])
    df_feature = df[features]
    index = df.index

    test_indices = np.random.RandomState(random_state).choice(index, size=int((len(index)) * test_size), replace=False)

    df_feature_train = df_feature[~df.index.isin(test_indices)]
    df_target_train = df_label[~df.index.isin(test_indices)]

    df_feature_test = df_feature.iloc[test_indices, :]
    df_target_test = df_label.iloc[test_indices, :]

    return df_feature_train, df_feature_test, df_target_train, df_target_test
