import pandas as pd
import matplotlib.pyplot as plt
from DataProcessing.CleaningTools.impute import impute_mean


df = pd.read_csv('../data.csv', index_col=0)
# impute all mean
features = ['Confirmed', 'GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
for feature in features:
    impute_mean(df, feature)
    plt.hist(df[feature], bins=25)
    print(feature)
    plt.show()

print(df['Population Density'].max())