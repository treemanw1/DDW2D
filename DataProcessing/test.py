import pandas as pd

ls = [1,2,3,4,5]
ls = pd.DataFrame(ls)
features = ['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']
print(ls.shape)
df = pd.read_csv('data.csv')
print(df['Deaths'].shape)
print(df[features].shape)
