#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'cash_to_assets_change'


#%%


# read data

# linked funda
funda_permno = pd.read_hdf(DIR + '/data/funda_permno.h5', key='data')

# fields that we need
funda_permno = funda_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyear', 'che', 'at'])


#%%


# calculate signals

result = pd.DataFrame()

fyears = sorted(funda_permno['fyear'].unique())
for fyear in fyears:  # calculate the asset growth of all the stocks in each fiscal year
    if fyear == fyears[0]:
        continue

    df = funda_permno[funda_permno['fyear'] == fyear]
    df_last_fyear = funda_permno[funda_permno['fyear'] == fyear-1]

    # filter
    df = df[df['at'] >= 5]  # total asset no less than 5 million

    df = df.copy()
    df_last_fyear = df_last_fyear.copy()
    df['signal'] = df['che'] / df['at']
    df_last_fyear['signal'] = df_last_fyear['che'] / df_last_fyear['at']

    # signal is che/at in 'fyear' - che/at in 'fyear-1'
    df_join = df.join(
        df_last_fyear[['gvkey', 'signal']].set_index('gvkey'), on='gvkey', rsuffix='_last_fyear')
    df_join['diff'] = df_join['signal'] - df_join['signal_last_fyear']
    df_join.dropna(subset=['diff'], inplace=True)  # drop nan

    df_join = df_join[['gvkey', 'lpermno', 'datadate', 'fyear', 'diff']]  # fields that we need
    result = result.append(df_join, ignore_index=True)  # append to signal container

result.to_csv(DIR + f'/anomaly-project/{NAME}/signals.csv')
