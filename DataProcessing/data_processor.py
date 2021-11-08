import pandas as pd
from DataProcessing.Imputation import impute as im
from DataProcessing.DataScaling import data_scaling as scale

raw_df = pd.read_csv('data.csv', index_col=0)
# Imputation (Impute mean first, cross validate other methods later)
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
for feature in features:
    im.impute_mean(raw_df, feature)

# Data Scaling
normally_distributed = ['gini', 'Urban Population', 'Median Age']
not_normally_distributed = ['GDP_2018', 'Infant Mortality Rate', 'Population Density']
for feature in normally_distributed:
    scale.standardize(raw_df, feature)
for feature in not_normally_distributed:
    scale.normalize(raw_df, feature)

raw_df.to_csv('data_processed.csv', index=False)
