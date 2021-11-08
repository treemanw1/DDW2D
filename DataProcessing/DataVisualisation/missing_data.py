import pandas as pd

# Show number of NaN values in df
df = pd.read_csv('../data.csv')
print(df.isna().sum())
print(len(df.index))

# Determine if there are duplicate rows (by "Name")
print('Duplicate Values? ', df.duplicated(subset=['Country']).unique())