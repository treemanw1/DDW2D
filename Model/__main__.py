import numpy as np
import pandas as pd
import cohort_functions as cf

df = pd.read_csv('../DataProcessing/data_processed.csv')
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
no_features = len(features)

df_features, df_target = cf.get_features_targets(df, features, ['Deaths'])

# split the data into training and test data sets
df_feature_train, df_feature_test, df_target_train, df_target_test = cf.split_data(df_features, df_target, random_state=100, test_size=0.3)

X = df_feature_train
X = cf.prepare_feature(X)
y = cf.prepare_target(df_target_train)

iterations = 1500
alpha = 0.01
beta = np.zeros((no_features+1, 1))

# call the gradient_descent function
beta, J_storage = cf.gradient_descent(X, y, beta, alpha, iterations)

# call the predict method to get the predicted values
pred = cf.predict(df_feature_test, beta)

target = cf.prepare_target(df_target_test)

r2 = cf.r2_score(target, pred)
print(r2)
mse = cf.mean_squared_error(target, pred)
print(mse)