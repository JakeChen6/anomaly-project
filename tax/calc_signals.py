#!/usr/bin/env python
# coding: utf-8

#%%

import pandas as pd

DIR = '/Users/zhishe/myProjects/anomaly'

NAME = 'tax'


#%%


# read data

# linked funda
funda_permno = pd.read_hdf(DIR + '/data/funda_permno.h5', key='data')

# fields that we need
funda_permno = funda_permno.reindex(
    columns=['gvkey', 'lpermno', 'datadate', 'fyear',
             'ib', 'txfed', 'txfo', 'txt', 'txdi'])


#%%


# calculate signals

result = funda_permno[funda_permno.fyear >= 1973].copy()

# fyear >= 1973 and fyear <= 1978: 0.48
rows = result[(result.fyear >= 1973) & (result.fyear <= 1978)]
signals = (rows['txfed'] + rows['txfo']) * 0.48 / rows['ib']
signals_backup = (rows['txt'] - rows['txdi']) * 0.48 / rows['ib']
signals.fillna(signals_backup, inplace=True)
result.loc[rows.index, 'signal'] = signals

# fyear >= 1979 and fyear <= 1986: 0.46
rows = result[(result.fyear >= 1979) & (result.fyear <= 1986)]
signals = (rows['txfed'] + rows['txfo']) * 0.46 / rows['ib']
signals_backup = (rows['txt'] - rows['txdi']) * 0.46 / rows['ib']
signals.fillna(signals_backup, inplace=True)
result.loc[rows.index, 'signal'] = signals

# fyear == 1987
rows = result[result.fyear == 1987]
signals = (rows['txfed'] + rows['txfo']) * 0.4 / rows['ib']
signals_backup = (rows['txt'] - rows['txdi']) * 0.4 / rows['ib']
signals.fillna(signals_backup, inplace=True)
result.loc[rows.index, 'signal'] = signals

# fyear >= 1988 and fyear <= 1992
rows = result[(result.fyear >= 1988) & (result.fyear <= 1992)]
signals = (rows['txfed'] + rows['txfo']) * 0.34 / rows['ib']
signals_backup = (rows['txt'] - rows['txdi']) * 0.34 / rows['ib']
signals.fillna(signals_backup, inplace=True)
result.loc[rows.index, 'signal'] = signals

# fyear >= 1993
rows = result[result.fyear >= 1993]
signals = (rows['txfed'] + rows['txfo']) * 0.35 / rows['ib']
signals_backup = (rows['txt'] - rows['txdi']) * 0.35 / rows['ib']
signals.fillna(signals_backup, inplace=True)
result.loc[rows.index, 'signal'] = signals

result.dropna(subset=['signal'], inplace=True)

result.to_csv(DIR + f'/anomaly-project/{NAME}/signals.csv')
