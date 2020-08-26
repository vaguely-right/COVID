import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas_bokeh
pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)
mpl.rcParams['figure.figsize'] = [12.0,8.0]
sns.set_style("whitegrid")

#%% Get the Johns Hopkins US data
infile = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
us = pd.read_csv(infile)
us.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'],axis=1,inplace=True)
us.rename(columns={'Province_State':'State'},inplace=True)
us = us.groupby('State').sum()
us = us.transpose()
us.drop(['American Samoa','Diamond Princess','Grand Princess','Guam','Northern Mariana Islands','Virgin Islands'],axis=1,inplace=True)
us.replace(0,np.nan,inplace=True)


#%% Find the states with the largest daily increases
us.diff(7).div(7).max().sort_values(ascending=False)[0:10].index.tolist()

# How about the ones with the largest increase in the increase
us.diff(7).div(7).tail(2).diff().max().sort_values(ascending=False)
states = us.diff(7).div(7).tail(2).diff().max().sort_values(ascending=False).index[0:10].tolist()

#%% Plot some stuff
us[states].diff(7).div(7).plot_bokeh.line(figsize=(1200,800))
