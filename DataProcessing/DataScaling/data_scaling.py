import pandas as pd


def normalize(df, feature):
    min = df[feature].min()
    max = df[feature].max()
    df[feature] = df[feature].apply(lambda x: (x-min)/(max-min))


def standardize(df, feature):
    mean = df[feature].mean()
    std = df[feature].std()
    df[feature] = df[feature].apply(lambda x: (x-mean)/std)

#
# df = pd.read_csv('../data.csv', index_col=0)
# normally_distributed = ['gini', 'Urban Population', 'Median Age']
# not_normally_distributed = ['GDP_2018', 'Infant Mortality Rate', 'Population Density']
# for feature in normally_distributed:
#     standardize(df, feature)
# for feature in not_normally_distributed:
#     normalize(df, feature)
