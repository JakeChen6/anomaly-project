#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'gross_profitability'


#%%


# read data

# linked funda
funda_permno = pd.read_hdf(DIR + '/data/funda_permno.h5', key='data')

# fields that we need
funda_permno = funda_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyear',
             'at', 'revt', 'cogs'])


#%%


# calculate signals

result = funda_permno.copy()
result['gp'] = (result['revt'] - result['cogs']) / result['at']
result.dropna(subset=['gp'], inplace=True)

result.to_csv(DIR + f'/anomaly-project/{NAME}/signals.csv')
