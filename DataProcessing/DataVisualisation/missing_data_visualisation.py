import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def show_missing(df, plot=bool):
    # Show number of NaN values in df
    df = df[['GDP_2018', 'gini', 'Infant Mortality Rate', 'Population Density', 'Urban Population', 'Median Age']]
    missing_df = df.isna().sum()
    missing_df = pd.DataFrame({'Feature': missing_df.index, 'Missing count': missing_df.values})
    print(missing_df)

    if plot:
        # Barplot showing no. of missing values
        sns.barplot(data=missing_df, y='Missing count', x='Feature')
        plt.show()


# df = pd.read_csv('../data.csv', index_col=0)
# show_missing(df, True)
