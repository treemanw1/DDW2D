import numpy as np
import pandas as pd


# This python files loads all the feature/label csvs into one single dataframe
# load all csv files into separate dataframes
covid = pd.read_csv('datasets/covid_deaths.csv')
gdp = pd.read_csv('datasets/gdp.csv')
gini = pd.read_csv('datasets/wiid_gini.csv', low_memory=False)
inf = pd.read_csv('datasets/infant_mortality_rate.csv', encoding="ISO-8859-1")
age = pd.read_csv('datasets/median_age.csv')
pop = pd.read_csv('datasets/population_by_country_2020.csv')

# covid
# sum all covid deaths from 2020-01-22 to 2020-07-27 (by country)
covid = covid.rename(columns={"Country/Region": "Country"})
covid = covid[['Country', 'Confirmed', 'Deaths', 'Recovered']]
covid = covid.groupby(['Country']).sum()
countries = list(covid.index)
print('Countries:', list(countries))
print('')
print('COVID DATAFRAME:\n', covid)
print('\n\n')

# gdp
gdp = gdp[['Country ', '2018']]  # Index 2018 GDP only
gdp = gdp.rename(columns={"Country ": "Country", '2018' : 'GDP_2018'})
gdp = gdp[gdp['Country'].isin(countries)]
print('GDP DATAFRAME:\n', gdp)
print('\n\n')

# extract cols below from wiid_gini.csv
gini_year_cutoff = 2010
cols = ['id', 'country', 'year', ' gini ']
gini = gini[cols]
print('No. of unique countries:', len(gini['country'].unique()))
print('No. of unique countries w >{} data:'.format(gini_year_cutoff),
      len(gini[gini['year'] >= gini_year_cutoff]['country'].unique()))
all_countries = pd.DataFrame(gini['country'].unique())
countries_gini = pd.DataFrame(gini[gini['year'] >= gini_year_cutoff]['country'].unique())
missing_countries = pd.concat([all_countries, countries_gini]).drop_duplicates(keep=False)
print('Countries wo data:', len(missing_countries), '\n')
# print(missing_countries)
gini = gini[gini['year'] >= gini_year_cutoff]
gini = gini.groupby(['country']).mean()
gini = gini[gini.index.isin(countries)]
gini = pd.DataFrame(gini[' gini '])
gini = gini.rename(columns={' gini ' : 'gini'})
print('GINI DATAFRAME:\n', gini,'\n')

# infant mortality
inf = inf[(inf['Year'] == 2019) & (inf['Gender']=='Total')]
inf = inf[inf['Country'].isin(countries)]
inf = inf[['Country', 'Infant Mortality Rate']]
print('INFANT MORTALITY DATAFRAME:\n', inf)
print('\n\n')

# population
pop = pop[['Country (or dependency)', 'Density (P/Km²)', 'Urban Pop %', 'Med. Age']]
pop = pop.rename(columns={"Country (or dependency)": "Country", 'Density (P/Km²)':'Population Density',
                            'Urban Pop %':'Urban Population', 'Med. Age':'Median Age'})
pop = pop[pop['Country'].isin(countries)]
pop = pop.sort_values(by=['Country'])
print('POPULATION DATAFRAME:\n', pop)
print('\n\n')

# Combine all dataframes into single dataframe
df = covid.merge(gdp, how='left', left_on='Country', right_on='Country')
df = df.merge(gini, how='left', left_on='Country', right_index=True)
df = df.merge(inf, how='left', left_on='Country', right_on='Country')
df = df.merge(pop, how='left', left_on='Country', right_on='Country')

# Fix Urban Population Column
# Replace N.A. with NaN
df['Urban Population'] = df['Urban Population'].replace('N.A.', np.NaN)
df['Median Age'] = df['Median Age'].replace('N.A.', np.NaN)
# Convert 25 % to 0.25
df.loc[df['Urban Population'].notna(), 'Urban Population'] = df[df['Urban Population'].notna()]['Urban Population'].apply(lambda x: int(x[:-2])/100)
df.to_csv('data.csv')
