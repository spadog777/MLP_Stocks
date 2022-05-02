import os
import csv
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import matplotlib.dates as mdates
import pandas_datareader.data as web
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, max_error, explained_variance_score
from math import sqrt
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)

style.use('ggplot')

company_info = {
    ### real estate
    'AMT': 'American Tower',
    'SPG': 'Simon Property Group',
    'PLD': 'Prologis',
    'CCI': 'Crown Castle',
    ### Retail companies
    'WMT': 'Walmart Inc.',
    'KR': 'Kroger Retail',
    'TGT': 'Target Reteail',
    'COST': 'Costco Retail',
    'HD': 'Home Depot',
    ### others
    'NKTR': 'Nektar Therapeutics',
    'AMD': 'AMD',
    'JPM': 'JPMorgan Chase',
    'BAC': 'Bank of America',
    'KO': 'The Coca-Cola',
    'VET': 'Vermilion Energy',
    ### less than 10 years
    'TWTR': 'Twitter Inc.',
}

def get_df(symbol, test=False):
    if test:
        csv_path = 'csv/{}_test.csv'.format(symbol)
        start = dt.datetime(2020, 1, 1)
        end = dt.datetime(2020, 3, 30)
    else:
        csv_path = 'csv/{}_training.csv'.format(symbol)
        start = dt.datetime(2010, 1, 1)
        end = dt.datetime(2019, 12, 31)

    if not os.path.isfile(csv_path):
        web.DataReader(symbol, 'yahoo', start, end).to_csv(csv_path)

    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    #df['avg'] = (df['Open'] + df['Close']) / 2
    df['returns'] = (df['Close']/df['Close'].shift(1)) - 1
    df['month'] = df.index.month
    return df

def line_chart(company_symbols, output_name):
    for symbol in company_symbols:
        df = get_df(symbol)
        name = company_info[symbol]
        df['Open'].plot(label=name)

    title_name = '{} Stocks'.format(output_name)
    plt.title(title_name)
    plt.ylabel('Price')
    plt.legend()
    fig_path = 'png/{}_LineChart.png'.format(output_name)
    plt.savefig(fig_path)
    #plt.show()
    plt.clf()

def scatter_correlation(company_symbols, output_name):
    all_df = []
    all_names = []
    for symbol in company_symbols:
        df = get_df(symbol)['Open']
        all_df.append(df)
        name = company_info[symbol]
        all_names.append(name)

    stocks_comp = pd.concat(all_df, axis=1)
    stocks_comp.columns = all_names
    scatter_matrix(stocks_comp, figsize=(10,10), hist_kwds={'bins': 50})
    #title_name = '{} Scatter Correlation'.format(output_name)
    #plt.title(title_name)
    fig_path = 'png/{}_ScatterCorrelation.png'.format(output_name)
    plt.savefig(fig_path)
    plt.clf()

def histogram(company_symbols, output_name):
    for symbol in company_symbols:
        df = get_df(symbol)
        name = company_info[symbol]
        df['returns'].hist(bins=100, label=name, alpha=0.4, figsize=(12,6))

    title_name = '{} Histogram'.format(output_name)
    plt.title(title_name)
    plt.xlabel('Daily Percentage Change')
    plt.ylabel('days')
    plt.legend()
    fig_path = 'png/{}_histogram.png'.format(output_name)
    plt.savefig(fig_path)
    #plt.show()
    plt.clf()


def kde(company_symbols, output_name):
    for symbol in company_symbols:
        df = get_df(symbol)
        name = company_info[symbol]
        df['returns'].plot(kind='kde', label=name)

    title_name = '{} KDE(Kernel Density Estimation)'.format(output_name)
    plt.title(title_name)
    plt.xlabel('Daily Percentage Change')
    plt.ylabel('Density')
    plt.legend()
    fig_path = 'png/{}_KDE.png'.format(output_name)
    plt.savefig(fig_path)
    #plt.show()
    plt.clf()

def boxplot(symbol):
    df = get_df(symbol)
    df.boxplot(column='Open', by='month')
    name = company_info[symbol]
    plt.title(name)
    #plt.xlabel('Month')
    plt.ylabel('Price')
    fig_path = 'png/{}_boxplot.png'.format(symbol)
    plt.savefig(fig_path)
    plt.clf()

def acf_seasonal(symbol):
    df = get_df(symbol)
    # ACF
    name = company_info[symbol]
    acf_title = '{} ACF'.format(name)
    acf_plt = plot_acf(df['Open'], title=acf_title, lags=365)
    acf_plt.savefig('png/{}_ACF.png'.format(symbol))
    plt.clf()

    # seasonal decompose
    season_plt = seasonal_decompose(df['Open'], model='additive', period=365)
    season_plt.plot()
    plt.savefig('png/{}_seasonal.png'.format(symbol))
    plt.clf()

def arima(symbol, p, d, q):
    train_ori = get_df(symbol)
    df = train_ori[['Open']].copy()
    train = df.Open
    test_ori = get_df(symbol, test=True)
    df = test_ori[['Open']].copy()
    test = df.Open
    model_arima = ARIMA(train, order=(p, d, q))
    model_arima_fit = model_arima.fit(disp=False)

    fc, se, conf = model_arima_fit.forecast(steps=61)
    fc = pd.Series(fc, index=test.index)
    lower = pd.Series(conf[:, 0], index=test.index)
    upper = pd.Series(conf[:, 1], index=test.index)
    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter('%b')
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    plt.plot(test, label='actual')
    plt.plot(fc, label='forecast')
    plt.fill_between(lower.index, lower, upper, color='k', alpha=0.1)
    name = company_info[symbol]
    plt.title('{} Stock Price Prediction'.format(name))
    plt.legend()
    plt.savefig('png/{}_ARIMA({}-{}-{}).png'.format(symbol, p, d, q))
    plt.clf()
    return
    '''
    actual = test#[:period]
    print(actual)
    print(predictions)
    plt.plot(actual)
    plt.plot(predictions, color='red')
    plt.savefig('png/{}_ARIMA({}-{}-{}).png'.format(symbol, p, d, q))
    mse = round(mean_squared_error(actual, predictions), 3)  
    rmse = round(sqrt(mse), 3)  
    r2 = round(r2_score(actual, predictions), 3)  
    mae = round(mean_absolute_error(actual, predictions), 3)  
    med_abs_err = round(median_absolute_error(actual, predictions, multioutput='raw_values')[0], 3)  
    max_err = round(max_error(actual, predictions), 3)  
    res_stdev = round(residual_stdev(actual, predictions), 3)

    target_file = 'prediction_results.csv'
    if not os.path.isfile(target_file):
        with open(target_file, 'a+', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(['Company Symbol', 'P', 'D', 'Q', 'MSE', 'RMSE', 'R2', 'MAE', 'MedAE', 'MaxErr', 'Residual Stdev'])

    with open(target_file, 'a+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
        row_format = [symbol, p, d, q, mse, rmse, r2, mae, med_abs_err, max_err, res_stdev]
        spamwriter.writerow(row_format)
    '''

def arima_tuning(symbol):
    try:
        arima(symbol, 4, 1, 4)
    except Exception as e:
       pass

    try:
        arima(symbol, 5, 1, 5)
    except Exception as e:
       pass

    return

    ### tuning test
    '''
    p_range = [0, 1, 2, 3, 4, 5, 6]
    d_range = [0, 1]
    q_range = [0, 1, 2, 3, 4, 5, 6]
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    arima(symbol, p, d, q)
                except (ValueError, np.linalg.LinAlgError) as e:
                    continue
    '''

##### group companies
grp_name = 'MISC'
grp = ['KO', 'AMD', 'WMT', 'VET']
line_chart(grp, grp_name)
scatter_correlation(grp, grp_name)
kde(grp, grp_name)
histogram(grp, grp_name)
for i in grp:
    boxplot(i)
    acf_seasonal(i)


grp_name = 'Retail Companies'
grp = ['WMT', 'KR', 'TGT', 'COST', 'HD']
line_chart(grp, grp_name)
scatter_correlation(grp, grp_name)
kde(grp, grp_name)
histogram(grp, grp_name)

for i in grp:
    boxplot(i)
    acf_seasonal(i)

grp_name = 'Real_Estate'
grp = ['AMT', 'SPG', 'PLD', 'CCI']
line_chart(grp, grp_name)
scatter_correlation(grp, grp_name)
kde(grp, grp_name)
histogram(grp, grp_name)

for i in grp:
    boxplot(i)
    acf_seasonal(i)


##### single company
for symbol, name in company_info.items():
    arima_tuning(symbol)
    boxplot(symbol)
    acf_seasonal(symbol)
