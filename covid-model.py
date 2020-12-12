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

#Function for getting the Alberta data
def get_ab_data():
    # Get the data
    url = r'https://www.alberta.ca/stats/covid-19-alberta-statistics.htm'
    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(url).text
    # Parse the html content
    soup = BeautifulSoup(html_content, "lxml")
    # Extract the data
    data = soup.find_all('script',attrs={'type':'application/json'})
    # Extract each data series from the json
    xnames = []
    ynames = []
    xdata = []
    ydata = []
    datnames = []
    for d in data:
        dat = json.loads(d.text)
        if('layout' in dat['x'].keys()):
            xname = dat['x']['layout']['xaxis']['title']
            yname = dat['x']['layout']['yaxis']['title']
            for i in dat['x']['data']:
                if 'name' in i.keys():
                    datname = i['name']
                else:
                    datname = ''
                xdat = i['x']
                ydat = i['y']
                xnames.append(xname)
                ynames.append(yname)
                datnames.append(datname)
                xdata.append(xdat)
                ydata.append(ydat)
    # Put the data together into a dataframe
    df = pd.concat([
            pd.DataFrame(index=xdata[6],data={'cumulative_cases':ydata[6]}),
            pd.DataFrame(index=xdata[7],data={'current_active':ydata[7]}),
            pd.DataFrame(index=xdata[8],data={'cumulative_recovered':ydata[8]}),
            pd.DataFrame(index=xdata[9],data={'cumulative_deaths':ydata[9]}),
            pd.DataFrame(index=xdata[13],data={'daily_confirmed':ydata[13]}),
            pd.DataFrame(index=xdata[14],data={'daily_probable':ydata[14]}),
            pd.DataFrame(index=xdata[15],data={'current_icu':ydata[15]}),
            pd.DataFrame(index=xdata[16],data={'current_non_icu':ydata[16]}),
            pd.DataFrame(index=xdata[20],data={'rolling_hospitalized':[i*4.37 for i in ydata[20]]}),
            pd.DataFrame(index=xdata[32],data={'daily_tested':ydata[32]}),
            pd.DataFrame(index=xdata[33],data={'daily_tests':ydata[33]}),
            ],axis=1).sort_index().fillna(0)
    # Need to calculate daily cases, daily deaths, current hospitalized, daily positivity
    df['daily_cases'] = df.daily_confirmed + df.daily_probable
    df['daily_deaths'] = df.cumulative_deaths.diff()
    df['current_hospitalized'] = df.current_icu + df.current_non_icu
    df['daily_positivity'] = df.daily_cases / df.daily_tests
    df = df[df.cumulative_cases>0]
    df.index = pd.to_datetime(df.index)
    return df

def get_ab_cases():
        # Get the data
    url = r'https://www.alberta.ca/stats/covid-19-alberta-statistics.htm'
    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(url).text
    # Parse the html content
    soup = BeautifulSoup(html_content, "lxml")
    # Extract the data
    data = soup.find_all('script',attrs={'type':'application/json'})
    dat = json.loads(data[20].text)
    ind = dat['x']['data'][0]
    date = dat['x']['data'][1]
    zone = dat['x']['data'][2]
    gender = dat['x']['data'][3]
    age = dat['x']['data'][4]
    status = dat['x']['data'][5]
    df = pd.DataFrame(index=ind,
                      data={'date':date,
                            'zone':zone,
                            'gender':gender,
                            'age':age,
                            'status':status})
    return df


#Function for getting the Alberta data, minus testing
def get_ab_data_notest():
    # Get the data
    url = r'https://www.alberta.ca/stats/covid-19-alberta-statistics.htm'
    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(url).text
    # Parse the html content
    soup = BeautifulSoup(html_content, "lxml")
    # Extract the data
    data = soup.find_all('script',attrs={'type':'application/json'})
    # Extract each data series from the json
    xnames = []
    ynames = []
    xdata = []
    ydata = []
    datnames = []
    for d in data:
        dat = json.loads(d.text)
        if('layout' in dat['x'].keys()):
            xname = dat['x']['layout']['xaxis']['title']
            yname = dat['x']['layout']['yaxis']['title']
            for i in dat['x']['data']:
                if 'name' in i.keys():
                    datname = i['name']
                else:
                    datname = ''
                xdat = i['x']
                ydat = i['y']
                xnames.append(xname)
                ynames.append(yname)
                datnames.append(datname)
                xdata.append(xdat)
                ydata.append(ydat)
    # Put the data together into a dataframe
    df = pd.concat([
            pd.DataFrame(index=xdata[6],data={'cumulative_cases':ydata[6]}),
            pd.DataFrame(index=xdata[7],data={'current_active':ydata[7]}),
            pd.DataFrame(index=xdata[8],data={'cumulative_recovered':ydata[8]}),
            pd.DataFrame(index=xdata[9],data={'cumulative_deaths':ydata[9]}),
            pd.DataFrame(index=xdata[13],data={'daily_confirmed':ydata[13]}),
            pd.DataFrame(index=xdata[14],data={'daily_probable':ydata[14]}),
            pd.DataFrame(index=xdata[15],data={'current_icu':ydata[15]}),
            pd.DataFrame(index=xdata[16],data={'current_non_icu':ydata[16]}),
            pd.DataFrame(index=xdata[20],data={'rolling_hospitalized':[i*4.37 for i in ydata[20]]}),
            ],axis=1).sort_index().fillna(0)
    # Need to calculate daily cases, daily deaths, current hospitalized, daily positivity
    df['daily_cases'] = df.daily_confirmed + df.daily_probable
    df['daily_deaths'] = df.cumulative_deaths.diff()
    df['current_hospitalized'] = df.current_icu + df.current_non_icu
    df = df[df.cumulative_cases>0]
    df.index = pd.to_datetime(df.index)
    return df

#Function for getting Canadian data
def get_can_data():
    #Data locations
    actfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/active_timeseries_prov.csv'
    casesfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/cases_timeseries_prov.csv'
    mortfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/mortality_timeseries_prov.csv'
    recfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/recovered_timeseries_prov.csv'
    testsfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/testing_timeseries_prov.csv'
    #Read the files
    active = pd.read_csv(actfile)
    cases = pd.read_csv(casesfile)
    deaths = pd.read_csv(mortfile)
    recoveries = pd.read_csv(recfile)
    tests = pd.read_csv(testsfile)
    #Standardize "date" as a name
    active.rename(columns={'date_active':'date'},inplace=True)
    cases.rename(columns={'date_report':'date'},inplace=True)
    deaths.rename(columns={'date_death_report':'date'},inplace=True)
    recoveries.rename(columns={'date_recovered':'date'},inplace=True)
    tests.rename(columns={'date_testing':'date'},inplace=True)
    #Combine the data sources
    indata = [cases,tests,deaths,active,recoveries]
    df = reduce(lambda left,right: pd.merge(left,right,on=['province','date'],how='outer',suffixes=[None,'_x']),indata)
    df.drop([i for i in df.columns.tolist() if '_x' in i],axis=1,inplace=True)
    df = df[df.province!='Repatriated']
    df['date'] = pd.to_datetime(df.date,dayfirst=True)
    df = df[df.date>='2020-03-01']
    #Fix the weekend zeros
    df['a'] = (df.cases.shift()!=0).cumsum()
    df = df.merge(df.groupby('a').mean().cases,left_on='a',right_index=True,suffixes=['_x',''])
    df['a'] = (df.testing.shift()!=0).cumsum()
    df = df.merge(df.groupby('a').mean().testing,left_on='a',right_index=True,suffixes=['_x',''])
    df = df[['province','date','cases','testing','deaths','cumulative_cases','cumulative_deaths','active_cases','recovered']]
    df.set_index(['province','date'],inplace=True)
#    df.cases.replace(0,np.nan,inplace=True)
#    df.dropna(inplace=True)
    return df

def fit_piecewise(y,n):
    x = np.array(range(len(y)))
    #Fit piecewise linear
    func = pwlf.PiecewiseLinFit(x,y)
    func.fit(n)
    #Predict
    yHat = func.predict(x)
    #Plot
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(x, yHat, '-')
    plt.show()
    return func,yHat

# Make a prediction of the next day based on a log-linear trend
def make_pred(y):
    Y = np.log(y)
    X = range(len(y))
    n = len(y)+1
    p = np.polyfit(X,Y,1)
    pred = np.exp(np.polyval(p,n))
    return pred

#%% Scrape, model, fit, and plot the Alberta data
df = get_ab_data()

y = np.log(df.daily_cases.replace(0,0.1).to_numpy())
cases_model,cases_fit = fit_piecewise(y,6)
df['fit_cases'] = np.exp(cases_fit)

x = np.array(range(len(y),len(y)+7))
d = df.index.min()+pd.to_timedelta(x,unit='D')
yHat = cases_model.predict(x)
proj = np.exp(yHat)
yVar = cases_model.prediction_variance(x)
mse = np.mean( (np.log(df.iloc[round(cases_model.fit_breaks[-2]):].fit_cases) - np.log(df.iloc[round(cases_model.fit_breaks[-2]):].daily_cases) ) **2 )
y5 = yHat - 1.65*np.sqrt(yVar+mse)
y95 = yHat + 1.65*np.sqrt(yVar+mse)
p5 = np.exp(y5)
p95 = np.exp(y95)


projdf = pd.DataFrame(index=d,data={'fit_cases':proj,'p5':p5,'p95':p95})
df = df.append(projdf)

df[['daily_cases','fit_cases','p5','p95']].loc['2020-09-10':].plot_bokeh(legend='top_left',figsize=(1200,800),number_format='0',plot_data_points=True)

df[['daily_cases','fit_cases','p5','p95']].plot(logy=True)
df[['daily_cases','fit_cases','p5','p95']].loc['2020-09-01':].plot()








