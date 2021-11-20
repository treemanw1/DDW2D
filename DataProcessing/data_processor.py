import pandas as pd
from DataProcessing.CleaningTools import impute as im
from DataProcessing.CleaningTools import outlier_removal as outlier
from DataProcessing.DataScaling import data_scaling as scale

raw_df = pd.read_csv('data.csv', index_col=0)
# Imputation (Impute mean first, cross validate other methods later)
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age', 'Confirmed']
for feature in features:
    im.impute_mean(raw_df, feature)

# Remove outliers
for feature in features:
    raw_df = outlier.remove_outliers_std(raw_df, feature, 2)
# 2 stds: 187 -> 155

# Scale Features
normally_distributed = ['gini', 'Urban Population', 'Median Age']
not_normally_distributed = ['GDP_2018', 'Infant Mortality Rate', 'Population Density', 'Confirmed']
for feature in normally_distributed:
    scale.standardize(raw_df, feature)
for feature in not_normally_distributed:
    scale.normalize(raw_df, feature)

# Scale Labels (standardize)
scale.standardize(raw_df, 'Deaths')

raw_df.to_csv('data_processed.csv', index=False)
