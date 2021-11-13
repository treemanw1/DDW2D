import pandas as pd
import numpy as np
import Model.cohort_functions as cf
import pearsons_correlation


df = pd.read_csv('../data_processed.csv')
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
no_features = len(features)

for feature in features:
    print(feature, ':', pearsons_correlation.pearsons_r(df, feature, 'Deaths'))