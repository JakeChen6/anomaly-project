#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'total_xfin'


#%%


# read data

# linked funda
funda_permno = pd.read_hdf(DIR + '/data/funda_permno.h5', key='data')

# fields that we need
funda_permno = funda_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyear',
             'at', 'sstk', 'dv', 'prstkc', 'dltis', 'dltr'])


#%%


# calculate signals

result = funda_permno.copy()
result['xfin'] = ((result['sstk'] - result['dv'] - result['prstkc']
                  + result['dltis'] - result['dltr']) / result['at'])
result.dropna(subset=['xfin'], inplace=True)

result.to_csv(DIR + f'/anomaly-project/{NAME}/signals.csv')
