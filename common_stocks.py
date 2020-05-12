#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:58:54 2020

@author: zhishe
"""

import time
import pickle as pk
from functools import reduce

import pandas as pd

ENV_PATH = '/Users/zhishe/myProjects/anomaly'

COMMON_STOCK_CODE = [10, 11]

# read data
MSEALL = pd.read_hdf(ENV_PATH + '/data/mseall.h5', key='mseall')


COMMON_STOCKS = dict()

start = time.time()
months = MSEALL['DATE'].unique()
months.sort()
for m in months:
    t = time.time()
    data = MSEALL[MSEALL['DATE'] <= m]  # no future information
    data = data.sort_values('DATE')
    data = data.drop_duplicates(subset=['PERMNO'], keep='last')
    # filter out non-common stocks
    conds = (data['SHRCD'] == c for c in COMMON_STOCK_CODE)  # conditions
    cond = reduce(lambda x, y: x | y, conds)
    data = data[cond]
    COMMON_STOCKS[m] = data['PERMNO'].unique().tolist()  # record the PERMNOs
    print(f'{m} done.', time.time() - t)
print(time.time() - start)


# save to local
with open(ENV_PATH + '/data/common_stock_permno.pkl', 'wb') as f:
    pk.dump(COMMON_STOCKS, f)
