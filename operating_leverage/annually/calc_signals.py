#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'operating_leverage'


#%%


# read data

# linked funda
funda_permno = pd.read_hdf(DIR + '/data/funda_permno.h5', key='data')

# fields that we need
funda_permno = funda_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyear', 'cogs', 'xsga', 'at'])


#%%


# calculate signals

result = funda_permno.copy()
result['ol'] = (result['cogs'] + result['xsga']) / result['at']
result.dropna(subset=['ol'], inplace=True)

result.to_csv(DIR + f'/anomaly-project/{NAME}/annually/signals.csv')
