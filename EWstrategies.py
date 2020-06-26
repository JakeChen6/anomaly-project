#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:40:44 2020

@author: zhishe
"""


#autocorrelation
#correlation between strategies (can we have diversification benefit)
#below are all market based anomalies, homogeneous!
#same category dimensionality reduction -> different categories combination.

import os
import pickle as pk

import numpy as np
import pandas as pd

DIR = DIR = '/Users/zhishe/myProjects/anomaly'

INCLUDE = ['momentum', 'lagged_momentum', 'industry_momentum', '52_week_high',
           'short_term_reversal', 'momentum_reversal', 'long_term_reversal',
           'seasonality', 'net_share_issuance', 'composite_equity_issuance',
           'stock_split', 'continuing_overreaction', 'asset_growth',
           'momentum_volume', 'firm_age_momentum', 'momentum_and_long_term_reversal']

combinations = [(12, 3), (12.7, 1), (6, 6), (12, 6),
                (1, 1), (18.13, 6), (60.13, 1),
                (240, 1), (12, 12), (60, 12),
                (1, 12), (12, 3), ('-', '-'),
                (6, 6), (11, 1), (12, 3)]

settings = dict(zip(INCLUDE, combinations))

result = pd.DataFrame()

for name, setting in settings.items():
    if name == 'asset_growth':
        with open(DIR + f'/anomaly-project/{name}/annually/monthly_returns.pkl', 'rb') as f:
            collector = pk.load(f)
        df = pd.Series(collector['ls'], name=name)
    else:
        with open(DIR + f'/anomaly-project/{name}/returns/monthly_returns.pkl', 'rb') as f:
            collector = pk.load(f)
        if name == 'stock_split':
            df = pd.Series(collector[setting]['w'], name=name)
        else:
            df = pd.Series(collector[setting]['ls'], name=name)
    result = pd.concat([result, df], axis=1)

result