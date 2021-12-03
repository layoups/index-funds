from logging import error
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
    if df.index[0] < datetime.strptime(start, "%Y-%m-%d") or df.index[-1] > datetime.strptime(end, "%Y-%m-%d"):
        return error

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
        try:
            universe = load_ticker_data([tickers[0]], start=start, end=end)
        except:
            universe = load_ticker_data(['AAPL'], start=start, end=end)

    for ticker in tickers[1:]:
        try:
            temp = get_ticker_data([ticker], start=start, end=end)
        except:
            try:
                temp = load_ticker_data([ticker], start=start, end=end)
            except:
                continue

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

def get_closest_trading_day(date, df):
    formatted_date = datetime.strptime(date, "%Y-%m-%d") if type(date) == str else date
    all_dates = df.index.get_level_values("date")
    date_index = np.argmin(
        np.abs(
            all_dates - formatted_date
        )
    )

    return all_dates[date_index]

##################### MODELS #####################

def get_portfolio_returns(index_weights, date, df):
    start_date = date
    end_date = start_date + relativedelta(months=3)

    relevant_returns = df[index_weights.T.index].reindex(
        pd.date_range(start_date, end_date)
    ).dropna().add(1)

    total_relevant_returns = relevant_returns.cumprod().iloc[-1]
    portfolio_returns = total_relevant_returns.multiply(index_weights).sum()

    return portfolio_returns

def get_spy_returns(date):
    start_date = date
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

def clustering_model(rolling_correlations, date, K, objective=GRB.MAXIMIZE):
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
        objective
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

def mean_variance_model(
    market_caps, 
    df, 
    date, 
    rolling_covariances, 
    center_weights,
    min_beta,
    max_beta,
    min_expected_residual_return
):
    date = str(date)
    benchmark_weights = market_caps.loc[(slice(None), date), :] /\
        market_caps.loc[(slice(None), date), :].sum()

    theData = {
        i: [df.loc[(i, date), :].values[0][-2], df.loc[(i, date), :].values[0][-1]]\
            for i in df.index.get_level_values(0).unique()
    }

    tickers, alphas, betas = gp.multidict(theData)
    betas, alphas = pd.Series(betas), pd.Series(alphas)

    covariances = rolling_covariances.loc[(date, slice(None)), :].values

    m = gp.Model("MeanVariance")
    x = pd.Series(
        m.addVars(
            tickers, 
            vtype = GRB.CONTINUOUS, 
            lb = 0, 
            ub = 1, 
            name = "tickers"
        )
    )

    objective = covariances.dot(
        x.subtract(benchmark_weights.reset_index(level=1, drop=True).MarketCap)
    ).dot(x.subtract(benchmark_weights.reset_index(level=1, drop=True).MarketCap))

    m.setObjective(
        objective, 
        GRB.MINIMIZE
    )

    only_centers = m.addConstrs(
        (x[i] == 0 for i in x.drop(center_weights.index).index), 
        name = "only_centers"
    )

    equal_one = m.addConstr(
        x.sum() == 1, 
        name = "equal_one"
    )

    beta_lower_bound = m.addConstr(
        x.dot(betas) >= min_beta, 
        name = "beta_lower_bound"
    )

    beta_upper_bound = m.addConstr(
        x.dot(betas) <= max_beta, 
        name = "beta_upper_bound"
    )

    alpha_min = m.addConstr(
        x.dot(alphas) >= min_expected_residual_return, 
        name = "alpha_min"
    )

    m.write("MeanVar.lp")
    m.optimize()

    x_results = pd.DataFrame(
        pd.Series(
            {i: x[i].x for i in x.to_dict()}
        ), 
        columns = ["weights"]
    )
    x_results.index.name = "center"

    opti_objective = np.sqrt(objective.getValue())

    return x_results, opti_objective

def compare_index_to_market(center_weights, date, ticker_data, ticker_data_wide):
    portfolio_returns = get_portfolio_returns(
        center_weights, date, ticker_data_wide
    )
    spy_returns = get_spy_returns(date)
    
    portfolio_beta = get_portfolio_beta(center_weights, date, ticker_data)

    return portfolio_returns, spy_returns, portfolio_returns - spy_returns, portfolio_beta

def master_func(
    date,
    K,
    rolling_correlations,
    market_caps, 
    ticker_data, 
    ticker_data_wide,
    rolling_covariances, 
    min_beta,
    max_beta,
    min_expected_residual_return,
    master_cluster_index,
    master_cluster_performance,
    master_mean_var_index,
    master_mean_var_performance,
    cluster_objective=GRB.MAXIMIZE
):
    start_date = get_closest_trading_day(date, ticker_data)
    try:
        _, _, z = clustering_model(rolling_correlations, date, K, cluster_objective)

        z_market_cap = market_caps.loc[
            market_caps.index.get_level_values(1) == date
        ].join(z)
        z_market_cap.reset_index(drop=True, level=1, inplace=True)

        center_weights = z_market_cap[
            (z_market_cap.in_center == 1) & (z_market_cap.is_center == 1)
        ].groupby("center").MarketCap.sum() /\
            z_market_cap[
                (z_market_cap.in_center == 1) & (z_market_cap.is_center == 1)
            ].MarketCap.sum()

        portfolio_returns, spy_returns, return_diff, portfolio_beta = \
            compare_index_to_market(center_weights, start_date, ticker_data, ticker_data_wide)

        for x in center_weights.index:
            master_cluster_index[(start_date, x)] = {'weight': center_weights.loc[x]}

        master_cluster_performance[start_date] = {
            "Index Returns": portfolio_returns,
            "SPY Returns": spy_returns,
            "Return Diff": return_diff,
            "Index Beta": portfolio_beta,
        }
        
        mean_var_step, obj = mean_variance_model(
            market_caps, 
            ticker_data, 
            date, 
            rolling_covariances, 
            center_weights,
            min_beta, 
            max_beta, 
            min_expected_residual_return
        )

        portfolio_returns, spy_returns, return_diff, portfolio_beta = \
            compare_index_to_market(
                mean_var_step.weights, 
                start_date, 
                ticker_data, 
                ticker_data_wide
            )

        for x in mean_var_step[mean_var_step.weights > 0].index:
            master_mean_var_index[(start_date, x)] = {
                'weight': mean_var_step[mean_var_step.weights > 0].loc[x].weights
            }

        master_mean_var_performance[start_date] = {
            "Index Returns": portfolio_returns,
            "SPY Returns": spy_returns,
            "Return Diff": return_diff,
            "Index Beta": portfolio_beta,
            "Active Risk": obj
        }

    except:
        None

    return True

    