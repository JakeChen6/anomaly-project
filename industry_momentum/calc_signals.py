import os
import time
import datetime as dt
import pickle as pk
from functools import reduce
from concurrent import futures

import psutil
import numpy as np
import pandas as pd

ENV_PATH = '/Users/zhishe/myProjects/anomaly'

NAME = 'industry_momentum'


#%%

# Filter

# common stocks only
#COMMON_STOCKS = [10, 11]

# exclude penny stocks
PRICE_LIMIT = 5

# listed on NYSE, AMEX, or NASDAQ
EXCH_CODE = [1, 2, 3]


#%%

# SIC codes
SIC = {'Mining': list(range(10, 15)),
       'Food': 20,
       'Apparel': [22, 23],
       'Paper': 26,
       'Chemical': 28,
       'Petroleum': 29,
       'Construction': 32,
       'Prim. Metals': 33,
       'Fab. Metals': 34,
       'Machinery': 35,
       'Electrical Eq.': 36,
       'Transport Eq.': 37,
       'Manufacturing': [38, 39],
       'Railroads': 40,
       'Other Transport': list(range(41, 48)),
       'Utilities': 49,
       'Dept. Stores': 53,
       'Retail': list(range(50, 53)) + list(range(54, 60)),
       'Financial': list(range(60, 70)),
       'Other': 'other'
       }

# ordered industry names
INDUSTRY = [
    'Mining',
    'Food',
    'Apparel',
    'Paper',
    'Chemical',
    'Petroleum',
    'Construction',
    'Prim. Metals',
    'Fab. Metals',
    'Machinery',
    'Electrical Eq.',
    'Transport Eq.',
    'Manufacturing',
    'Railroads',
    'Other Transport',
    'Utilities',
    'Dept. Stores',
    'Retail',
    'Financial',
    'Other'
    ]


#%%

# Read data

# monthly stock data
MSF = pd.read_hdf(ENV_PATH + '/data/msf.h5', key='msf')

# common stocks - PERMNOs
with open(ENV_PATH + '/data/common_stock_permno.pkl', 'rb') as f:
    COMMON_STOCK_PERMNO = pk.load(f)

# Transform HSICCD to two-digit codes
MSF['HSICCD'] //= 10  # three-digit -> two-digit, four-digit -> three-digit
index = MSF[MSF['HSICCD'] >= 100].index
MSF.loc[index, 'HSICCD'] //= 10  # three-digit -> two-digit

# range of dates
DATE_RANGE = MSF['DATE'].unique()
DATE_RANGE.sort()


#%%

# Apply filter

# exclude penny stocks
MSF = MSF[MSF['PRC'].abs() >= PRICE_LIMIT].copy()

# listed on NYSE, AMEX, or NASDAQ
conds = (MSF['HEXCD'] == c for c in EXCH_CODE)
cond = reduce(lambda x, y: x | y, conds)
MSF = MSF[cond].copy()


#%%

# Define signal calculation process

# signal: for each stock, its signal is the value-weighted return, over the past
# lb months, of the industry that the stock belongs to.


def get_weights(data):
    data = data.reset_index()
    data.sort_values(by='DATE', inplace=True)
    data = data.drop_duplicates(subset=['PERMNO'], keep='last')
    data.set_index('PERMNO', inplace=True)
    data['mktcap'] = data['PRC'].abs() * data['SHROUT']
    data['weight'] = data['mktcap'] / data['mktcap'].sum()

    return data['weight']


def calc_past_industry_rets(m, lb):
    """
    m:      current month
    lb:     look back period
    """
    # SELECT data
    start, end = DATE_RANGE[DATE_RANGE < m][[-lb, -1]]
    data = MSF[
        (MSF['DATE'] >= start) &
        (MSF['DATE'] <= end)
    ]

    # only include common stocks
    common_stock_permno = COMMON_STOCK_PERMNO[end]

    data = data.set_index('PERMNO')
    index = set(data.index) & set(common_stock_permno)
    data = data.loc[index]  # only keep common stocks

    # calculate each industry's value-weighted return
    ind_rets = {}

    for ind in INDUSTRY:
        if ind != 'Other':
            siccd = SIC[ind]

            # SELECT data WHERE HSICCD is siccd or in siccd
            if isinstance(siccd, list):
                conds = (data['HSICCD'] == c for c in siccd)
                cond = reduce(lambda x, y: x | y, conds)
            else:
                cond = data['HSICCD'] == siccd

            subdata = data[cond].copy()
            data.drop(subdata.index, inplace=True)  # what are left finally are stocks in industry 'Other'
        else:
            subdata = data

        # cumulative return in the past lb months
        cum_rets = (subdata['RET'] + 1).groupby(level=0).prod(min_count=1)
        cum_rets.dropna(inplace=True)  # drop NaN
        # value weight this industry
        subdata = subdata.loc[cum_rets.index]
        weights = get_weights(subdata)
        val_weighted_ret = (cum_rets * weights).sum()
        ind_rets[ind] = {s: val_weighted_ret for s in cum_rets.index}

    # save in a pandas.Series
    stock_ind_rets = pd.Series(dtype=np.float64)
    for ind, d in ind_rets.items():
        stock_ind_rets = stock_ind_rets.append(pd.Series(d))

    stock_ind_rets.name = 'RET'
    stock_ind_rets.index.name = 'PERMNO'

    return stock_ind_rets


def get_signals(args):
    """
    pno:            process identifier
    subrange:       subset of the date_range
    lb:             look back period
    """
    subrange, lb = args

    # signal: the cumulative return over the past lb months.
    signals = {m: calc_past_industry_rets(m, lb) for m in subrange}

    return signals


#%%

# Distribute workloads to multiple CPUs

CPU_COUNT = psutil.cpu_count(logical=False)

COLLECTOR = {}  # the container where we store the computation results

START = 1963
END   = 2018

# look back periods & holding periods
LOOK_BACK = [6]
HOLDING   = [6]

# we only specify starting and ending years above, here we infer the starting
# and ending months.
START = DATE_RANGE[DATE_RANGE >= np.datetime64(dt.date(START, 7, 1))][0]
END = DATE_RANGE[DATE_RANGE <= np.datetime64(dt.date(END, 12, 31))][-1]


for lb in LOOK_BACK:
    hd = max(HOLDING)
    print('\nCalculating (%s, %s) strategy...' % (lb, hd), end='\t')

    # on this date we calculate the first set of signals
    first_date = DATE_RANGE[DATE_RANGE <= START][-hd]
    # calculate signals for every month in this range
    date_range = DATE_RANGE[(DATE_RANGE >= first_date) & (DATE_RANGE <= END)]

    # we will split the task and distribute the workloads to multiple processes
    size = len(date_range) // CPU_COUNT  # the size of each workload
    chunks = []
    for i in range(CPU_COUNT):
        if i != CPU_COUNT-1:
            chunks.append(date_range[size*i:size*(i+1)])
        else:
            chunks.append(date_range[size*i:])

    with futures.ProcessPoolExecutor(max_workers=CPU_COUNT) as ex:
        start_time = time.time()
        res = ex.map(get_signals, zip(chunks, [lb] * CPU_COUNT))
        for signals in res:
            COLLECTOR.setdefault(lb, {}).update(signals)
        print('{:.2f} s.'.format(time.time() - start_time))


#%%
# Output

if not os.path.exists(ENV_PATH + f'/results/{NAME}'):
    os.mkdir(ENV_PATH + f'/results/{NAME}')
    os.mkdir(ENV_PATH + f'/results/{NAME}/signals')

for lb in LOOK_BACK:
    table = pd.DataFrame()
    signals = COLLECTOR[lb]
    # consolidate each month's signals into a single table
    for k, v in signals.items():
        df = pd.DataFrame(v)
        df.reset_index(inplace=True)
        df['DATE'] = k
        table = pd.concat([table, df], ignore_index=True)

    table.rename(columns={'RET': 'SIGNAL'}, inplace=True)
    table = table.reindex(columns=['DATE', 'PERMNO', 'SIGNAL'])  # reorder the columns
    table.sort_values(by='DATE', inplace=True)  # sort by 'DATE'
    table.to_csv(ENV_PATH + f'/results/{NAME}/signals/{lb}.csv')  # write to a CSV
    print('%s done.' % lb)
