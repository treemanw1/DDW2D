import numpy as np
import pandas as pd
import cohort_functions as cf
import nevilles_functions as wee

df = pd.read_csv('../DataProcessing/data_processed.csv')
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
no_features = len(features)

df_features, df_target = cf.get_features_targets(df, features, ['Deaths'])

r2, mse = wee.cross_val_score(df_features, df_target, beta=np.zeros((no_features+1, 1)),
                              alpha=1, num_iters=10000, random_state=100, cv=10)
print(r2, mse)
