#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'operating_leverage'


#%%


# read data

# linked fundq
fundq_permno = pd.read_hdf(DIR + '/data/fundq_permno.h5')
fundq_permno.rename(columns={'LPERMNO': 'lpermno'}, inplace=True)

# fields that we need
fundq_permno = fundq_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyearq', 'fqtr', 'fyr', 'rdq', 'cogsq', 'xsgaq', 'atq'])


#%%


# calculate signals

result = fundq_permno.copy()
result['ol'] = (result['cogsq'] + result['xsgaq']) / result['atq']
result.dropna(subset=['ol'], inplace=True)

result.to_csv(DIR + f'/anomaly-project/{NAME}/quarterly/signals.csv')
