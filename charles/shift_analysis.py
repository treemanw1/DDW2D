import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('datasets/owid-covid-data.csv')
df = df[df['continent'].notna()]
df['new_cases'] = df['new_cases'].fillna(0)
df['new_deaths'] = df['new_deaths'].fillna(0)

iso_codes = df['iso_code'].to_numpy()
iso_codes = np.unique(iso_codes)


def assert_continuity(sub_df):
    dates = pd.to_datetime(sub_df['date'])
    date_diffs = dates.diff().to_numpy()[1:]
    second_diffs = date_diffs / np.timedelta64(1, 's')
    day_diffs = second_diffs / (24 * 3600)
    assert max(day_diffs) < 2


def covariances(shift_start=0, shift_end=20):
    for shift in range(shift_start, shift_end):
        for iso_code in iso_codes:
            iso_df = df[df['iso_code'] == iso_code]
            assert_continuity(iso_df)

