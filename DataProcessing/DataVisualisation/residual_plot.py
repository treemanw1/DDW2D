import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataProcessing.CleaningTools.impute import impute_mean
import Model.cohort_functions as cf

# Plots actual vs predicted values
df = pd.read_csv('../data_processed.csv', index_col=0)
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
no_features = len(features)

df_feature = df[features]
df_target = pd.DataFrame(df['Deaths'])

df_feature_train, df_feature_test, df_target_train, df_target_test = cf.split_data(df_feature, df_target, test_size=0.3)

X = df_feature_train
X = cf.prepare_feature(X)
y = cf.prepare_target(df_target_train)

beta, J_storage = cf.gradient_descent(X, y, beta=np.zeros((no_features+1, 1)), alpha=0.01, num_iters=10000)

df_target_pred = cf.predict(df_feature_test, beta)
df_target_pred = pd.DataFrame(df_target_pred)
df_target_pred.index = df_target_test.index
df_target_pred = df_target_pred.rename({0: 'Predictions'}, axis='columns')

df_plot = pd.concat([df_target_test, df_target_pred], axis=1)
print(df_plot)
sns.scatterplot(data=df_plot, x="Deaths", y='Predictions')
plt.show()
