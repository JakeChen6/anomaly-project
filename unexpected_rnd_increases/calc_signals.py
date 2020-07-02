#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd
pd.set_option('display.max_columns', None)

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'unexpected_rnd_increases'


#%%


# read data

# linked funda
funda_permno = pd.read_hdf(DIR + '/data/funda_permno.h5', key='data')

# fields that we need
funda_permno = funda_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyear',
             'at', 'xrd', 'sale'])


#%%


# calculate signals

result = pd.DataFrame()

fyears = sorted(funda_permno['fyear'].unique())
for fyear in fyears:
    if fyear == fyears[0]:
        continue

    df = funda_permno[funda_permno['fyear'] == fyear].copy()
    df_last_year = funda_permno[funda_permno['fyear'] == fyear-1].copy()

    # filter
    df = df[df['at'] > 0]  # total assets greater than 0
    df = df[df['sale'] != 0]  # revenue not equal to 0

    df['XRD/SALE'] = df['xrd'] / df['sale']  # R&D scaled by revenue
    df['XRD/AT'] = df['xrd'] / df['at']  # R&D scaled by assets
    df_last_year['XRD/AT'] = df_last_year['xrd'] / df_last_year['at']  # R&D scaled by assets

    df_join = df.join(
        df_last_year[['gvkey', 'xrd', 'XRD/AT']].set_index('gvkey'), on='gvkey', rsuffix='_last_fyear')
    df_join['pct_chg_in_rnd'] = df_join['xrd'] / df_join['xrd_last_fyear'] - 1
    df_join['pct_chg_in_ratio'] = df_join['XRD/AT'] / df_join['XRD/AT_last_fyear'] - 1

    # drop nan
    df_join.dropna(subset=['XRD/SALE', 'XRD/AT', 'pct_chg_in_rnd', 'pct_chg_in_ratio'],
                   inplace=True)

    # R&D scaled by revenue and R&D scaled by assets > 0
    df_join = df_join[df_join['XRD/SALE'] > 0]
    df_join = df_join[df_join['XRD/AT'] > 0]

    # yearly percentage change in R&D expenditure > 5%
    df_join = df_join[df_join['pct_chg_in_rnd'] > 0.05]

    # R&D scaled by assets increases by more than 5%
    df_join = df_join[df_join['pct_chg_in_ratio'] > 0.05]

    result = result.append(df_join, ignore_index=True)

result.to_csv(DIR + f'/anomaly-project/{NAME}/signals.csv')
