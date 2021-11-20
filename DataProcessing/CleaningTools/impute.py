# Imputation methods for missing values


def impute_mean(df, feature):
    mean = df[feature].mean()
    df[feature] = df[feature].fillna(mean)
    # return df


def impute_mode(df, feature):
    mode = df[feature].mode()
    df[feature] = df[feature].fillna(mode)
    # return df


def impute_median(df, feature):
    median = df[feature].median()
    df[feature] = df[feature].fillna(median)
    # return df


def impute_constant(df, feature, const=0):
    df[feature] = df[feature].fillna(const)
    # return df


# # Read in df
# df = pd.read_csv('../data.csv', index_col=0)
# # show_missing(df, False)
# # impute_mean(df, 'GDP_2018')
# # show_missing(df, False)
# features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
# for feature in features:
#     print(type(df[feature][2]))
#     impute_mean(df, feature)
