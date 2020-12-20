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


#%% Use statsmodels to predict the next day with prediction intervals

def make_prediction(Y,output='mean',alph=0.1):
    if output not in ['mean','lower','upper','r_mean','r_lower','r_upper']:
        print('Output type not recognized, defaulting to mean')
        output = 'mean'
    y = np.log(Y.to_numpy())
#    x = np.transpose(np.array([np.ones(len(y)),range(len(y))]))
    x = sm.add_constant(np.array(range(len(y))))
    lm = sm.OLS(y,x).fit()
    if(output=='mean'):
        out = np.exp(lm.get_prediction(np.array([1,len(x)])).summary_frame(alpha=alph))['mean']
    elif(output=='lower'):
        out = np.exp(lm.get_prediction(np.array([1,len(x)])).summary_frame(alpha=alph))['obs_ci_lower']
    elif(output=='upper'):
        out = np.exp(lm.get_prediction(np.array([1,len(x)])).summary_frame(alpha=alph))['obs_ci_upper']
    elif(output=='r_mean'):
        out = np.exp(lm.params[1])
    elif(output=='r_lower'):
        out = np.exp(lm.conf_int(alpha=alph)[1,0])
    elif(output=='r_upper'):
        out = np.exp(lm.conf_int(alpha=alph)[1,1])
    return out

#%%
# Second-order model, projected forward linearly?
def make_second_order(Y,output='mean',alph=0.1):
    if output not in ['mean','lower','upper','r_mean','r_lower','r_upper']:
        print('Output type not recognized, defaulting to mean')
        output = 'mean'
    



#%%
df = get_ab_data()
#df = df.loc['2020-06-01':]    

#%% Look at the RMSE etc for the number of days
optn = []
for ndays in range(5,14):
    df['fit_cases'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'mean'}).shift(1)
    df['p5'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'lower'}).shift(1)
    df['p95'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'upper'}).shift(1)
#    df['slope'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'slope'}).shift(1)
    rmse = np.sqrt(((df.daily_cases-df.fit_cases)**2).mean())
    rmselog = np.sqrt(((np.log(df.daily_cases)-np.log(df.fit_cases))**2).sum())
    p5check = (df.daily_cases<df.p5).mean()
    p95check = (df.daily_cases<df.p95).mean()
    optn.append([ndays,rmse,rmselog,p5check,p95check])

pd.DataFrame(optn)

#%%
ndays = 11
df['fit_cases'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'mean'}).shift(1)
df['p5'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'lower'}).shift(1)
df['p95'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'upper'}).shift(1)
df['r'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'r_mean'})
df['r5'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'r_lower'})
df['r95'] = df.daily_cases.rolling(ndays).apply(make_prediction,kwargs={'output':'r_upper'})

df[['daily_cases','fit_cases','p5','p95']].plot(logy=True)
df[['daily_cases','fit_cases','p5','p95']].loc['2020-09-01':].plot()

df[['r','r5','r95']].plot()
df[['r','r5','r95']].loc['2020-09-01':].plot()

#%% Add a new day to the dataframe
mean = make_prediction(df.daily_cases.tail(ndays),output='mean').iloc[0]
lower = make_prediction(df.daily_cases.tail(ndays),output='lower').iloc[0]
upper = make_prediction(df.daily_cases.tail(ndays),output='upper').iloc[0]


df.loc[df.index.max()+pd.to_timedelta(1,unit='D')] = np.nan

df['fit_cases'].iloc[-1] = mean
df['p5'].iloc[-1] = lower
df['p95'].iloc[-1] = upper

df[['daily_cases','fit_cases','p5','p95']].plot(logy=True)
df[['daily_cases','fit_cases','p5','p95']].loc['2020-09-01':].plot()
