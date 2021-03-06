#!/usr/bin/env python
# coding: utf-8

#%%


from functools import lru_cache
import pickle as pk

import numpy as np
import pandas as pd
from scipy import stats as sci_stats

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PreText, Select, TextInput, Button
from bokeh.plotting import figure

DIR = '/Users/zhishe/myProjects/anomaly'


#%%


# anomaly setting

NAME = '52_week_high'

LOOK_BACK = ['12']
HOLDING = ['6']


#%%


# read data

with open(DIR + f'/anomaly-project/{NAME}/returns/monthly_returns.pkl', 'rb') as f:
    MONTHLY_RETS = pk.load(f)


#%%


@lru_cache()
def get_data(lb, hd):
    lb, hd = map(int, (lb, hd))
    monthly_rets = MONTHLY_RETS[(lb, hd)]
    data = pd.DataFrame(monthly_rets)
    data.index.name='date'
    data.sort_index(inplace=True)
    return data


def calc_netval_and_drawdown(data):
    data = data.copy()
    data['w_netval'] = (data['w'] + 1).cumprod()
    data['l_netval'] = (data['l'] + 1).cumprod()
    data['ls_netval'] = (data['ls'] + 1).cumprod()
    data['w_drawdown'] = (data['w_netval'] / data['w_netval'].cummax()) - 1
    data['l_drawdown'] = (data['l_netval'] / data['l_netval'].cummax()) - 1
    data['ls_drawdown'] = (data['ls_netval'] / data['ls_netval'].cummax()) - 1
    return data


#%%


title = PreText(text=NAME, width=500)

# set up widgets
stats = PreText(text='', width=500)  # the statistics table
lookback = Select(value=LOOK_BACK[0], options=LOOK_BACK)  # length of look back period
holding = Select(value=HOLDING[0], options=HOLDING)  # length of holding period
start = TextInput(value='', title='Start')  # start of sample
end = TextInput(value='', title='End')  # end of sample
button = Button(label='Change', button_type='primary')

# set up plots
source = ColumnDataSource(data=dict(date=[], w_netval=[], l_netval=[], ls_netval=[]))
tools = 'pan,wheel_zoom,xbox_select,reset'

# ts1
ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts1.line('date', 'w_netval', source=source)
ts1.yaxis.axis_label = 'Net Value'
ts1.circle('date', 'w_netval', size=1, source=source, color=None, selection_color="orange")

# ts2
ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts2.line('date', 'l_netval', source=source)
ts2.yaxis.axis_label = 'Net Value'
ts2.circle('date', 'l_netval', size=1, source=source, color=None, selection_color="orange")

# ts3
ts3 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
ts3.line('date', 'ls_netval', source=source)
ts3.yaxis.axis_label = 'Net Value'
ts3.circle('date', 'ls_netval', size=1, source=source, color=None, selection_color="orange")

ts1.title.text, ts2.title.text, ts3.title.text = 'Winner', 'Loser', 'Long-Short'

ts2.x_range = ts1.x_range
ts3.x_range = ts1.x_range


#%%


# set up callbacks

def lookback_change(attrname, old, new):
    start.value = ''  # refresh
    end.value = ''
    update()

def holding_change(attrname, old, new):
    start.value = ''  # refresh
    end.value = ''
    update()

def button_click():
    update(s=start.value, e=end.value)


def update(s=None, e=None):
    lb, hd = lookback.value, holding.value
    data = get_data(lb, hd)

    # assuming valid inputs below, i.e., both s and e are in range.
    if s:
        s = pd.Timestamp(s)
        data = data.loc[s:]
    if e:
        e = pd.Timestamp(e)
        data = data.loc[:e]

    data = calc_netval_and_drawdown(data)  # calculate net value and drawdown
    source.data = data[['w_netval', 'l_netval', 'ls_netval']]

    update_stats(data)


def update_stats(data):
    df = pd.DataFrame(columns=['w', 'l', 'ls'])
    rets = data[['w', 'l', 'ls']]
    df.loc['avg monthly ret'] = rets.mean(axis=0)
    df.loc['t-statistic'] = rets.apply(lambda x: sci_stats.ttest_1samp(x, 0).statistic, axis=0)
    df.loc['skew'] = rets.apply(lambda x: x.skew(), axis=0)  # skewness
    df.loc['kurtosis'] = rets.apply(lambda x: x.kurtosis(), axis=0)  # kurtosis
    df.loc['sharpe ratio'] = df.loc['avg monthly ret'] / rets.std(axis=0) * np.sqrt(12)  # scale to annual
    # calculate positive month %
    for col in df.columns:
        s = rets[col]
        df.loc['positive month %', col] = s[s > 0].shape[0] / s.shape[0]
    # calculate maximum drawdown
    for col in df.columns:
        df.loc['max drawdown', col] = data[f'{col}_drawdown'].min()
    
    stats.text = str(df)


def selection_change(attrname, old, new):
    lb, hd = lookback.value, holding.value
    data = get_data(lb, hd)
    selected = source.selected.indices
    if selected:
        data = data.iloc[selected, :]
        start.value = str(data.index[0].date())
        end.value = str(data.index[-1].date())
    data = calc_netval_and_drawdown(data)  # calculate net value and drawdown
    update_stats(data)


lookback.on_change('value', lookback_change)
holding.on_change('value', holding_change)
button.on_click(button_click)

source.selected.on_change('indices', selection_change)


#%%


# set up layout
widgets = column(lookback, holding, row(start, end), button, stats)
main_row = row(widgets)
series = column(ts1, ts2, ts3)
layout = column(title, main_row, series)

# initialize
update()

curdoc().add_root(layout)
curdoc().title = NAME

