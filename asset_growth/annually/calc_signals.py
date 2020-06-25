#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'asset_growth'


#%%


# read data

# linked funda
funda_permno = pd.read_hdf(DIR + '/data/funda_permno.h5', key='data')

# fields that we need
funda_permno = funda_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyear', 'at'])


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

    # signal is AT for 'fyear' scaled by AT for 'fyear-1' minus one
    df_join = df.join(
        df_last_fyear[['gvkey', 'at']].set_index('gvkey'), on='gvkey', rsuffix='_last_fyear')
    df_join['growth'] = df_join['at'] / df_join['at_last_fyear'] - 1  # ratio minus one
    df_join.dropna(subset=['growth'], inplace=True)  # drop nan

    df_join = df_join[['gvkey', 'lpermno', 'datadate', 'fyear', 'growth']]  # fields that we need
    result = result.append(df_join, ignore_index=True)  # append to signal container

result.to_csv(DIR + f'/anomaly-project/{NAME}/annually/signals.csv')
