#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'asset_growth'


#%%


# read data

# linked fundq
fundq_permno = pd.read_hdf(DIR + '/data/fundq_permno.h5')
fundq_permno.rename(columns={'LPERMNO': 'lpermno'}, inplace=True)

# fields that we need
fundq_permno = fundq_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyearq', 'fqtr', 'fyr', 'rdq', 'atq'])


#%%


# calculate signals

result = pd.DataFrame()

fyearqs = sorted(fundq_permno['fyearq'].unique())

for fyearq in fyearqs:
    if fyearq == fyearqs[0]:
        continue

    for fqtr in [1, 2, 3, 4]:
        df = fundq_permno[(fundq_permno['fyearq'] == fyearq) &
                          (fundq_permno['fqtr'] == fqtr)]
        df_lagged = fundq_permno[(fundq_permno['fyearq'] == fyearq-1) &
                                 (fundq_permno['fqtr'] == fqtr)]

        df = df.dropna(subset=['atq'])
        df_lagged = df_lagged.dropna(subset=['atq'])
        if df.empty or df_lagged.empty:
            continue

        # filter
        df = df[df['atq'] >= 5]  # total asset no less than 5 million
        
        # signal is ATQ for 'fyear' scaled by ATQ for 'fyear-1' minus one
        df_join = df.join(
            df_lagged[['gvkey', 'atq']].set_index('gvkey'), on='gvkey', rsuffix='_lagged')
        df_join['growth'] = df_join['atq'] / df_join['atq_lagged'] - 1  # ratio minus one
        df_join.dropna(subset=['growth'], inplace=True)  # drop nan

        df_join = df_join[  # fields that we need
            ['gvkey', 'lpermno', 'datadate', 'fyearq', 'fqtr', 'fyr', 'rdq', 'growth']]

        result = result.append(df_join, ignore_index=True)  # append to signal container

result.to_csv(DIR + f'/anomaly-project/{NAME}/quarterly/signals.csv')
