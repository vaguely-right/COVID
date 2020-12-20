import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pwlf
from functools import reduce
from scipy.optimize import differential_evolution,minimize
from bs4 import BeautifulSoup
import requests
import json
from datetime import date
from datetime import datetime
from datetime import timedelta
import pandas_bokeh
import statsmodels.api as sm

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)
mpl.rcParams['figure.figsize'] = [12.0,8.0]
sns.set_style("whitegrid")




#%% Look at individual cases a bit
df = get_ab_cases()

# Case fatality rate by age
df.pivot_table(index='age',columns='status',values='zone',aggfunc=len).apply(lambda x: x.Died/(x.Died+x.Recovered),axis=1)

# Infections by age by date
sns.lineplot(df.groupby(['date','age']).status.count().reset_index(),x='date',y='status',hue='age')

df.groupby(['date','age']).status.count().reset_index().plot.area()

df.pivot_table(index='date',columns='age',values='zone',aggfunc=len).plot.area()




#%% Trying out a model based on test positivity again
# Idea: daily test positives are missing a number of current unknown active infections
df = get_ab_data()

k = 8.8
alpha = 1/2
gamma = 1/6

#(np.exp(-k*df.daily_positivity).plot())

df['current_unknown'] = df.fit_cases / (np.exp(-k*df.daily_positivity.rolling(7).mean()))

df['delta_unknown'] = df.current_unknown.diff()

df['current_exposed'] = (df.delta_unknown + df.fit_cases + gamma * df.current_unknown.shift(1)) / alpha

df['delta_exposed'] = df.current_exposed.diff()

df['beta'] = (df.delta_exposed + alpha * df.current_exposed.shift(1)) / df.current_unknown.shift(1)
#df.daily_cases.corr(df.current_unknown.shift(-7))

#df.daily_cases.corr(df.current_unknown.shift(-i)) for i in range(28)]


#%% Trying an ARIMA model


df = get_ab_data()
df = df.loc['2020-03-15':]

X = df.index.values
 
Y = np.log(df.daily_cases.to_numpy())

mod = sm.tsa.ARIMA(Y,order=(7,0,7))

res = mod.fit()

print(res.summary())

res.plot_predict(1,len(Y)+30)

df['fit_cases'] = np.exp(res.predict())

df[['daily_cases','fit_cases']].plot()


#%% Functions from https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_stationarity(timeseries):
    # rolling statistics
    rolling_mean = timeseries.rolling(window=7).mean()
    rolling_std = timeseries.rolling(window=7).std()
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # Dickey–Fuller test:
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

#%%
df = get_ab_data()
df = df.loc['2020-03-15':]

timeseries = (np.log(df.daily_cases)-np.log(df.daily_cases).rolling(7).mean()).dropna()

get_stationarity(df.daily_cases)
get_stationarity(np.log(df.daily_cases))
get_stationarity(timeseries)














