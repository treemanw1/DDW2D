from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def assert_continuity(sub_df):
    dates = pd.to_datetime(sub_df['date'])
    date_diffs = dates.diff().to_numpy()[1:]
    second_diffs = date_diffs / np.timedelta64(1, 's')
    day_diffs = second_diffs / (24 * 3600)
    assert max(day_diffs) < 2


def covariances(
    base_df, shift_start=0, shift_end=50,
    cases_column='new_cases_smoothed',
    deaths_column='new_deaths_smoothed',
    stride=1
):
    iso_codes = base_df['iso_code'].to_numpy()
    iso_codes = np.unique(iso_codes)
    corr_mapping = {}

    for shift in tqdm(range(shift_start, shift_end)):
        all_cases, all_deaths = [], []

        for iso_code in iso_codes:
            iso_df = base_df[base_df['iso_code'] == iso_code]
            # assert_continuity(iso_df)

            cases = iso_df[cases_column].fillna(0).to_numpy()
            deaths = iso_df[deaths_column].fillna(0).to_numpy()

            length = len(cases)
            cases = cases[:length-shift]
            deaths = deaths[shift:]

            all_cases.append(cases)
            all_deaths.append(deaths)

        all_cases = np.concatenate(all_cases, axis=0)[::stride]
        all_deaths = np.concatenate(all_deaths, axis=0)[::stride]
        correlation = np.corrcoef(all_cases, all_deaths)
        # print(f'CORR {shift} {correlation}')
        corr_mapping[shift] = correlation

    return corr_mapping


if __name__ == '__main__':
    df = pd.read_csv('datasets/owid-covid-data.csv')
    df = df[df['continent'].notna()]
    df['new_cases'] = df['new_cases'].fillna(0)
    df['new_deaths'] = df['new_deaths'].fillna(0)
    covariances(base_df=df)

