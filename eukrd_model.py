import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pwlf
from functools import reduce
from scipy.optimize import differential_evolution,minimize

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)
mpl.rcParams['figure.figsize'] = [12.0,8.0]
sns.set_style("whitegrid")

#%% Function for going through the EUKRD model

def eukrd(beta,alpha,p,k,gamma,cfr,e_0,u_0,k_0,r_0,d_0):    
    d_e = beta * ( u_0 + k_0 ) - alpha * e_0
    d_u = alpha * e_0 - gamma * u_0 - np.exp(-k * p) * u_0
    d_k = np.exp(-k * p) * u+0 - gamma * k_0
    d_c[t] = np.exp(-k * p[t]) * i_u[t-1]
    d_r[t] = gamma * (i_u[t-1] + i_k[t-1] * (1-cfr[t]))
    d_d[t] = gamma * i_k[t-1] * cfr[t]
    e[t] = e[t-1] + d_e[t]
    i_u[t] = i_u[t-1] + d_i_u[t]
    i_k[t] = i_k[t-1] + d_i_k[t]
    r[t] = r[t-1] + d_r[t]
    d[t] = d[t-1] + d_d[t]

    return e_1,u_1,k_1,r_1,d_1,d_e,d_u,d_k,d_r,d_d


#%% Function for going through a (S)EIR model
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

#%% Get the Canadian data
actfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/active_timeseries_prov.csv'
casesfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/cases_timeseries_prov.csv'
mortfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/mortality_timeseries_prov.csv'
recfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/recovered_timeseries_prov.csv'
testsfile = r'https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/timeseries_prov/testing_timeseries_prov.csv'

active = pd.read_csv(actfile)
cases = pd.read_csv(casesfile)
deaths = pd.read_csv(mortfile)
recoveries = pd.read_csv(recfile)
tests = pd.read_csv(testsfile)

active.rename(columns={'date_active':'date'},inplace=True)
cases.rename(columns={'date_report':'date'},inplace=True)
deaths.rename(columns={'date_death_report':'date'},inplace=True)
recoveries.rename(columns={'date_recovered':'date'},inplace=True)
tests.rename(columns={'date_testing':'date'},inplace=True)

indata = [cases,tests,deaths,active,recoveries]

#df = cases.merge(tests,on=['province','date'])

df = reduce(lambda left,right: pd.merge(left,right,on=['province','date'],how='outer',suffixes=[None,'_x']),indata)
df.drop([i for i in df.columns.tolist() if '_x' in i],axis=1,inplace=True)

df = df[df.province=='Alberta']
df = df[['date','cases','testing','deaths','cumulative_cases','cumulative_deaths','active_cases','recovered']]
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#%% Fix the weekend zeros
df['a'] = (df.cases.shift()!=0).cumsum()
df = df.merge(df.groupby('a').mean().cases,left_on='a',right_index=True,suffixes=['_x',''])

df['a'] = (df.testing.shift()!=0).cumsum()
df = df.merge(df.groupby('a').mean().testing,left_on='a',right_index=True,suffixes=['_x',''])
df = df[['date','cases','testing','deaths','cumulative_cases','cumulative_deaths','active_cases','recovered']]

df.cases.replace(0,np.nan,inplace=True)
df.dropna(inplace=True)

#%% Calculate the positivity
df['test_positivity'] = df.cases / df.testing

# The first day is all tests to date; correct for that
df.test_positivity.iloc[0] = df.cumulative_cases.iloc[0] / df.testing.iloc[0]

#sns.lineplot(data=df,x='date',y='test_positivity')

#%% Determine the CFR values
#sns.lineplot(data=df,x='cumulative_cases',y='cumulative_deaths')
df.plot.scatter('cumulative_cases','cumulative_deaths')

x = df.cumulative_cases.to_numpy()
y = df.cumulative_deaths.to_numpy()

cfr_fit = pwlf.PiecewiseLinFit(x,y)
res = cfr_fit.fit(3)

# predict for the determined points
xHat = np.linspace(min(x), max(x), num=100)
yHat = cfr_fit.predict(xHat)

# plot the results
plt.figure()
plt.plot(x, y, 'o')
plt.plot(xHat, yHat, '-')
plt.show()

cfr = cfr_fit.slopes
df['cfr'] = np.sum([cfr[i]*((res[i]<=df.cumulative_cases) & (df.cumulative_cases<=res[i+1])) for i in range(3)],axis=0)


#%% Get a rough Rt value using seven-day rolling averages
df['rt'] = df.cases.rolling(14).mean().shift(-14)/df.cases.rolling(14).mean()
df.rt.fillna(method='bfill',inplace=True)
df.rt.fillna(method='ffill',inplace=True)



#%% Run through a SEIR model
alpha = 1/2
gamma = 1/12
beta = 1.0 * gamma

e_0 = df.iloc[0].cumulative_cases / 6
i_0 = df.iloc[0].cumulative_cases
r_0 = 0

e = [e_0]
i = [i_0]
r = [r_0]
d_e = [0]
d_i = [0]
d_r = [0]
d_c = [i_0]

for t in range(233):
    e_1,i_1,r_1,d_e_1,d_i_1,d_r_1 = seir(beta,alpha,gamma,e[t-1],i[t-1],r[t-1])
    e.append(e_1)
    i.append(i_1)
    r.append(r_1)
    d_e.append(d_e_1)
    d_i.append(d_i_1)
    d_r.append(d_r_1)
    d_c_1 = d_i_1 + d_r_1
    d_c.append(d_c_1)


#%% Optimize the beta parameter for 14-day windows in a SEIR model
alpha = 1/2
gamma = 1/12

e_0 = df.iloc[0].cumulative_cases * gamma / alpha
i_0 = df.iloc[0].cumulative_cases
r_0 = 0

ndays = 14

beta_opt = minimize(eukrd_err,x0=beta,args=(p,cfr,alpha,gamma,k,e_0,i_u_0,i_k_0,r_0,d_0,d_c_true),bounds=betarange)








