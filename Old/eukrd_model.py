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

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)
mpl.rcParams['figure.figsize'] = [12.0,8.0]
sns.set_style("whitegrid")

#%% Function for getting the Alberta data
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
            pd.DataFrame(index=xdata[32],data={'daily_tested':ydata[32]}),
            pd.DataFrame(index=xdata[33],data={'daily_tests':ydata[33]}),
            ],axis=1).sort_index().fillna(0)
    # Need to calculate daily cases, daily deaths, current hospitalized, daily positivity
    df['daily_cases'] = df.daily_confirmed + df.daily_probable
    df['daily_deaths'] = df.cumulative_deaths.diff()
    df['current_hospitalized'] = df.current_icu + df.current_non_icu
    df['daily_positivity'] = df.daily_cases / df.daily_tests
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

# Function for going through the EUKR model
def eukr(beta,alpha,p,k,gamma,e_0,u_0,k_0,r_0):    
    d_e = beta * ( u_0 + k_0 ) - alpha * e_0
    d_u = alpha * e_0 - gamma * u_0 - np.exp(-k * p) * u_0
    d_k = np.exp(-k * p) * u_0 - gamma * k_0
    d_c = np.exp(-k * p) * u_0
    d_r = gamma * (u_0 + k_0)
    e_1 = e_0 + d_e
    u_1 = u_0 + d_u
    k_1 = k_0 + d_k
    r_1 = r_0 + d_r
    return e_1,u_1,k_1,r_1,d_c,d_e,d_u,d_k,d_r


# Function for going through the EUKRD model

def eukrd(beta,alpha,p,k,gamma,cfr,e_0,u_0,k_0,r_0,d_0):    
    d_e = beta * ( u_0 + k_0 ) - alpha * e_0
    d_u = alpha * e_0 - gamma * u_0 - np.exp(-k * p) * u_0
    d_k = np.exp(-k * p) * u_0 - gamma * k_0
    d_c = np.exp(-k * p) * u_0
    d_r = gamma * (u_0 + k_0 * (1-cfr))
    d_d = gamma * k_0 * cfr
    e_1 = e_0 + d_e
    u_1 = u_0 + d_u
    k_1 = k_0 + d_k
    r_1 = r_0 + d_r
    d_1 = d_0 + d_d
    return e_1,u_1,k_1,r_1,d_1,d_c,d_e,d_u,d_k,d_r,d_d


# Function for going through a (S)EIR model
def seir(beta,alpha,gamma,e_0,i_0,r_0):
    d_e = beta * i_0 - alpha * e_0
    d_i = alpha * e_0 - gamma * i_0
    d_r = gamma * i_0
    e_1 = e_0 + d_e
    i_1 = i_0 + d_i
    r_1 = r_0 + d_r
    return e_1,i_1,r_1,d_e,d_i,d_r


# RMSE function for a SEIR model over n days
def seir_rmse(beta,alpha,gamma,e,i,r,ndays,true_cases):
    c = 0
    for t in range(ndays):
        e,i,r,d_e,d_i,d_r = seir(beta,alpha,gamma,e,i,r)
        c = c + d_i+d_r
    rmse = np.sqrt((c-true_cases)**2)
    return rmse


#%%
df = get_ab_data()    
df = df.loc['2020-03-01':]

#%% Get a rough Rt value using seven-day rolling averages
ndays = 7
df['rt'] = df.daily_cases.rolling(ndays).mean().shift(-ndays)/df.daily_cases.rolling(ndays).mean()
df.rt.fillna(method='bfill',inplace=True)
df.rt.fillna(method='ffill',inplace=True)
# Clip it to like 4?
#df.rt[df.rt>4] = 10

#%% Run through a SEIR model
alpha = 1/2
gamma = 1/12
beta = df.rt.to_numpy() * gamma

e_0 = 20
i_0 = 0
r_0 = 0

e = [e_0]
i = [i_0]
r = [r_0]
d_e = [0]
d_i = [0]
d_r = [0]
d_c = [i_0]

for t in range(233):
    e_1,i_1,r_1,d_e_1,d_i_1,d_r_1 = seir(beta[t-1],alpha,gamma,e[t-1],i[t-1],r[t-1])
    e.append(e_1)
    i.append(i_1)
    r.append(r_1)
    d_e.append(d_e_1)
    d_i.append(d_i_1)
    d_r.append(d_r_1)
    d_c_1 = d_i_1 + d_r_1
    d_c.append(d_c_1)

plt.plot(d_c)
plt.plot(df.daily_cases)

#%% Run through a EUKR model
alpha = 1/4
gamma = 1/10
beta = df.rt.to_numpy() * gamma

pk = 10
p = df.daily_positivity.to_numpy()

e_0 = 2
u_0 = 12
k_0 = 0
r_0 = 0
d_c_0 = 0

e = [e_0]
u = [u_0]
k = [k_0]
r = [r_0]
d_c = [d_c_0]
d_e = [0]
d_u =[0]
d_k =[0]
d_r = [0]


for t in range(248):
    e_1,u_1,k_1,r_1,d_c_1,d_e_1,d_u_1,d_k_1,d_r_1 = eukr(beta[t-1],alpha,p[t-1],pk,gamma,e[t-1],u[t-1],k[t-1],r[t-1])
    e.append(e_1)
    u.append(u_1)
    k.append(k_1)
    r.append(r_1)
    d_c.append(d_c_1)
    d_e.append(d_e_1)
    d_u.append(d_u_1)
    d_k.append(d_k_1)
    d_r.append(d_r_1)



plt.plot(d_c)
plt.plot(df.daily_cases)










