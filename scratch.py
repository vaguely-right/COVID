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

(np.exp(-k*df.daily_positivity).plot())

df['current_unknown'] = df.daily_cases / (np.exp(-k*df.daily_positivity))

df.daily_cases.corr(df.current_unknown.shift(-7))

[df.daily_cases.corr(df.current_unknown.shift(-i)) for i in range(28)]







