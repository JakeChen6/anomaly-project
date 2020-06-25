#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'cash_to_assets'


#%%


# read data

# linked fundq
fundq_permno = pd.read_hdf(DIR + '/data/fundq_permno.h5')
fundq_permno.rename(columns={'LPERMNO': 'lpermno'}, inplace=True)

# fields that we need
fundq_permno = fundq_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyearq', 'fqtr', 'fyr', 'rdq', 'cheq', 'atq'])


#%%


# calculate signals

result = fundq_permno.copy()
result['signal-cta'] = result['cheq'] / result['atq']
result.dropna(subset=['signal-cta'], inplace=True)

result.to_csv(DIR + f'/anomaly-project/{NAME}/signals.csv')
