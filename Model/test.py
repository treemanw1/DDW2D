import nevilles_functions as wee
import cohort_functions as cf
import pandas as pd
import numpy as np

# df = pd.read_csv('../DataProcessing/data_processed.csv')
# features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
# df_features, df_target = cf.get_features_targets(df, features, ['Deaths'])
# no_features = len(features)
#
# iterations = 1500
# alpha = 0.01
# beta = np.zeros((no_features+1, 1))
#
#
# r2, mse = wee.cross_val_score(df_features, df_target, beta, alpha, iterations)
# print(r2)
# print(mse)
