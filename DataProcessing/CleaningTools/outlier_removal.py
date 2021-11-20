# Tools for removing outliers
import pandas as pd


def remove_outliers_std(df, feature, number=3):
    std = df[feature].std()
    mean = df[feature].mean()
    return df[(df[feature] > mean-number*std) & (df[feature] < mean+number*std)]


# # Read in df
# df = pd.read_csv('../data.csv', index_col=0)
# print(len(df))
# features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
# for feature in features:
#     df = remove_outliers_std(df, feature)
# print(len(df))
