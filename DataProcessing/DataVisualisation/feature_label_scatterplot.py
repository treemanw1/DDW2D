import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataProcessing.CleaningTools.impute import impute_mean

# Plots feature against label values via a scatterplot
df = pd.read_csv('../data_processed.csv', index_col=0)
# impute all mean
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age', 'Confirmed']
# fig, axs = plt.subplots(ncols=len(features))
for i in range(len(features)):
    impute_mean(df, features[i])
    sns.scatterplot(data=df, x=features[i], y="Deaths")
    plt.show()
    print(features[i])

