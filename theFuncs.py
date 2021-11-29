import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.rolling import RollingOLS
from datetime import datetime
from dateutil.relativedelta import relativedelta

import quandl
import pandas_datareader.data as web

import gurobipy as gp
from gurobipy import GRB

import os

##################### DATA #####################

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
    df['ret'] = df.groupby('ticker').close.pct_change()
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
    df['ret'] = df.groupby('ticker').close.pct_change()
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
    spy_dtb3['spy_ret'] = spy_dtb3.spy_close.pct_change()
    return spy_dtb3

def multi_index_merge(df1, df2, index1, index2):
    temp = df1.join(df2)
    temp.set_index(index2, inplace=True, append=True)
    temp = temp.reorder_levels([index2, index1]).sort_index()
    # temp['ret_premium'] = temp['ret'] - temp['DTB3']
    # temp['spy_ret_premium'] = temp['spy_ret'] - temp['DTB3']
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
        pd.read_csv('./data/spy_dtb3.csv', index_col='date', parse_dates=True), 
        'date', 
        'ticker'
    )
    universe = clean_data(universe)
    # universe['Beta'] = 0
    # universe['alpha'] = 0
    return universe

##################### MODELS #####################

def get_portfolio_returns(index_weights, date, df):
    start_date = datetime.strptime(date, "%Y-%m-%d")
    end_date = start_date + relativedelta(months=3)

    relevant_returns = df[index_weights.T.index].reindex(
        pd.date_range(start_date, end_date)
    ).dropna().add(1)

    total_relevant_returns = relevant_returns.cumprod().iloc[-1]
    portfolio_returns = total_relevant_returns.multiply(index_weights).sum()

    return portfolio_returns

def get_spy_returns(date):
    start_date = datetime.strptime(date, "%Y-%m-%d")
    end_date = start_date + relativedelta(months=3)

    spy_returns = pd.read_csv(
        'data/spy_dtb3.csv', 
        index_col=0, 
        parse_dates=True
    ).reindex(
        pd.date_range(start_date, end_date)
    ).dropna().spy_ret.add(1).cumprod().iloc[-1]

    return spy_returns

def get_portfolio_beta(index_weights, date, df):
    return df.loc[
        (index_weights.index, date), 
        ["alpha", "Beta"]
    ].reset_index(
        level=1, 
        drop=True
    ).Beta.multiply(index_weights).sum()


##################### MODELS #####################

def clustering_model(rolling_correlations, date, K):
    tups = {}

    for i in rolling_correlations.columns:
        for j in rolling_correlations.columns:

            tups[(i, j)] = rolling_correlations.loc[(date, i), j]
            tups[(j, i)] = rolling_correlations.loc[(date, j), i]

    tickers, correlations = gp.multidict(tups)

    m = gp.Model("Clustering")

    x = m.addVars(
        tickers, 
        vtype = GRB.BINARY, 
        name = "X"
    )

    y = m.addVars(
        rolling_correlations.columns, 
        vtype = GRB.BINARY, 
        name = "Y"
    )

    portfolio_similarity = x.prod(correlations)
    m.setObjective(
        portfolio_similarity, 
        GRB.MAXIMIZE
    )

    cluster_numbers = m.addConstr(
        y.sum('*') == K, 
        name = "cluster_numbers"
    )

    one_ticker_one_cluster = m.addConstrs(
        (x.sum(i, '*') == 1 for i in y), 
        name = "one_ticker_one_cluster"
    )

    cluster_centers = m.addConstrs(
        (x[i, j] <= y[j] for i in y for j in y), 
        name = "cluster_centers"
    )

    m.write("Clustering.lp")
    m.optimize()

    y_results = pd.DataFrame(
        pd.Series({i: y[i].x for i in y}), 
        columns=["is_center"]
    )
    y_results.index.name = "center"

    x_results = pd.DataFrame(
        pd.Series({i: x[i].x for i in x}), 
        columns=["in_center"]
    ) 
    x_results.index.set_names(["ticker", "center"], inplace=True)

    return x_results, y_results, x_results.join(y_results)

def mean_variance_model():
    None