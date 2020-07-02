#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'profitability'


#%%


# read data

# linked fundq
fundq_permno = pd.read_hdf(DIR + '/data/fundq_permno.h5')
fundq_permno.rename(columns={'LPERMNO': 'lpermno'}, inplace=True)

# fields that we need
fundq_permno = fundq_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyearq', 'fqtr', 'fyr', 'rdq', 'ibq', 'atq'])


#%%


# calculate signals

result = pd.DataFrame()

fyearqs = sorted(fundq_permno['fyearq'].unique())

for fyearq in fyearqs:
    for fqtr in [1, 2, 3, 4]:
        if fqtr == 1:
            df_last_quarter = fundq_permno[(fundq_permno['fyearq'] == fyearq-1) &
                                           (fundq_permno['fqtr'] == 4)]
        else:
            df_last_quarter = fundq_permno[fundq_permno['fqtr'] == fqtr-1]

        if df_last_quarter.empty:
            continue

        df = fundq_permno[(fundq_permno['fyearq'] == fyearq) &
                          (fundq_permno['fqtr'] == fqtr)]

        df_join = df.join(
            df_last_quarter[['gvkey', 'atq']].set_index('gvkey'), on='gvkey', rsuffix='_last_qtr')
        df_join['signal'] = df_join['ibq'] / df_join['atq_last_qtr']
        df_join.dropna(subset=['signal'], inplace=True)  # drop nan

        result = result.append(df_join, ignore_index=True)  # append to signal container

result.to_csv(DIR + f'/anomaly-project/{NAME}/signals.csv')
