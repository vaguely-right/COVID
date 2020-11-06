#%%
from bs4 import BeautifulSoup
import requests
import json
import pandas as pd

pd.set_option('display.width',150)
pd.set_option('display.max_columns',16)

#%% Testing section
url = r'https://www.alberta.ca/stats/covid-19-alberta-statistics.htm'
# Make a GET request to fetch the raw HTML content
html_content = requests.get(url).text

# Parse the html content
soup = BeautifulSoup(html_content, "lxml")

# The div tag with attrs={class="section level2 unnumbered"} finds each tab in the website
# Extract the testing data
testing_data = soup.find_all('script',attrs={'type':'application/json'})

print(testing_data[6].prettify())

# Extract the JSON data
dat = json.loads(testing_data[6].text)

# These are the x and y labels:
dat['x']['layout']['xaxis']['title']
dat['x']['layout']['yaxis']['title']

# Data series names:
dat['x']['data'][0]['name']
[i['name'] for i in dat['x']['data']]

# This is the data:
dat['x']['data'][0]['x']
dat['x']['data'][0]['y']
[i['y'] for i in dat['x']['data']]


#%% Get the data
url = r'https://www.alberta.ca/stats/covid-19-alberta-statistics.htm'
# Make a GET request to fetch the raw HTML content
html_content = requests.get(url).text

# Parse the html content
soup = BeautifulSoup(html_content, "lxml")

# Extract the data
data = soup.find_all('script',attrs={'type':'application/json'})

#%% Extract each data series from the json

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
        
data_df = pd.DataFrame({'xaxis':xnames,
                        'yaxis':ynames,
                        'series':datnames,
                        'xdata':xdata,
                        'ydata':ydata})

#%% Put the data together into a dataframe
    
# The important series that I want:
# 6: Total cumulative cases
# 7: Active cases
# 8: Cumulative recovered cases
# 9: Cumulative deaths
# 13: Daily confirmed cases
# 14: Daily probable cases
# 15: Currently in ICU
# 16: Currently in hospital, non-ICU
# 32: Daily people tested
# 33: Daily tests

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

#%%
import pwlf
import numpy as np
import matplotlib.pyplot as plt

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







