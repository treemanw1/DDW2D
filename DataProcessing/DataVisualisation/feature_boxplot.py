import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataProcessing.CleaningTools.impute import impute_mean

# Visualize feature outliers via boxplot
df = pd.read_csv('../data.csv', index_col=0)
df = pd.read_csv('../data_processed.csv', index_col=0)
# impute all mean
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age', 'Confirmed']
# fig, axs = plt.subplots(ncols=len(features))
for i in range(len(features)):
    impute_mean(df, features[i])
    sns.boxplot(data=df, x=features[i])
    plt.show()
    print(features[i])
