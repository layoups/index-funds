import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.rolling import RollingOLS

import quandl
import pandas_datareader.data as web

import os

def load_ticker_data(tickers, start="2000-01-01", end="2021-11-12"):
    df = web.DataReader(tickers[0], 'yahoo', start=start, end=end)
    df['ticker'] = tickers[0]
    if len(tickers) > 1:
        for ticker in tickers[1:]:
            temp = web.DataReader(ticker, 'yahoo', start=start, end=end)
            temp['ticker'] = ticker
            df = pd.concat(
                [
                    df,
                    temp
                ]
            )
    df.columns = map(str.lower, df.columns)
    df.index.name = str.lower(df.index.name)
    df.drop(['adj close'], inplace=True, axis=1)
    return df

def get_ticker_data(tickers, start="2000-01-01", end="2021-11-12"):
    df = pd.read_csv(
            './data/stock_dfs/{}.csv'.format(tickers[0]), 
            parse_dates=True, 
            index_col=0
        )
    df['ticker'] = tickers[0]
    if len(tickers) > 1:
        for ticker in tickers[1:]:
            temp = pd.read_csv(
                './data/stock_dfs/{}.csv'.format(ticker), 
                parse_dates=True, 
                index_col=0
            )
            temp['ticker'] = ticker
            df = pd.concat(
                [
                    df,
                    temp
                ]
            )
    df.columns = map(str.lower, df.columns)
    df.index.name = str.lower(df.index.name)
    df.drop(['adj close'], inplace=True, axis=1)
    return df


def load_DTB3_SPY(start="2000-01-01", end="2021-11-12"):
    spy_dtb3 = pd.concat(
        [
            web.DataReader('SPY', 'yahoo', start=start, end=end)['Adj Close'],
            quandl.get('FRED/DTB3', start_date=start, end_date=end)
        ],
        axis=1
    )

    spy_dtb3.columns = ['spy_close', 'DTB3']
    spy_dtb3.index.rename('date', inplace=True)
    spy_dtb3['DTB3'] = spy_dtb3['DTB3'] / 100.0

    return spy_dtb3

def multi_index_merge(df1, df2, index1, index2):
    temp = df1.join(df2)
    temp.set_index(index2, inplace=True, append=True)
    temp = temp.reorder_levels([index2, index1]).sort_index()
    return temp

def clean_data(df, many=True):
    df.replace([-np.inf, np.inf], np.nan, inplace=True)
    # Forward fill to avoid look ahead bias
    if many:
        df = df.groupby('ticker').fillna(method='ffill')
    else:
        df = df.fillna(method='ffill')

    return df.dropna()

def get_ticker_data_multisource(tickers, start="2000-01-01", end="2021-11-12"):
    try:
        universe = get_ticker_data([tickers[0]], start=start, end=end)
    except:
        universe = load_ticker_data([tickers[0]], start=start, end=end)

    for ticker in tickers[1:]:
        try:
            temp = get_ticker_data([ticker], start=start, end=end)
        except:
            temp = load_ticker_data([ticker], start=start, end=end)

        universe = pd.concat(
            [
                universe, 
                temp
            ]
        )

    universe = multi_index_merge(
        universe,
        pd.read_csv('./data/spy_vix_dtb3.csv', index_col='date', parse_dates=True), 
        'date', 
        'ticker'
    )
    universe = clean_data(universe)
    return universe