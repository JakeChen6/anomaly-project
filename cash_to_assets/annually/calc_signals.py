#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'cash_to_assets'


#%%


# read data

# linked funda
funda_permno = pd.read_hdf(DIR + '/data/funda_permno.h5', key='data')

# fields that we need
funda_permno = funda_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyear', 'che', 'at'])


#%%



# calculate signals

result = funda_permno.copy()
result['signal-cta'] = result['che'] / result['at']
result.dropna(subset=['signal-cta'], inplace=True)

result.to_csv(DIR + f'/anomaly-project/{NAME}/annually/signals.csv')
