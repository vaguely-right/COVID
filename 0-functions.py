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

#%% Functions
# Calculate the slope of a relationship
def get_slope(Y):
    n = len(Y)
    X = range(0,n)
    p = np.polyfit(X,Y,1)
    return p[0]

# Make a prediction of the next day based on a linear trend
def make_pred(df):
    Y = df.dropna().to_numpy()
    X = df.dropna().index.values
    n = X.max()+1
    p = np.polyfit(X,Y,1)
    pred = np.polyval(p,n)
    return pred

# Return the r-squared relationship
def rsq(x,y):
    r = np.corrcoef(x,y)[1,0]**2
    return r

# Convert a dataframe to days since 100th case (90th)
def hundo(df,cols):
    frames = []
    for c in cols:
        frames.append(df[df[c]>=90].reset_index()[c].to_frame())
    h = pd.concat(frames,axis=1)
    h = h[h.max().sort_values(ascending=False).index]
    return h

# Return the smoothed derivative of a dataframe
def getd1(df,ndays):
    frames = []
    for c in df.columns:
        slope = np.log(df[c].dropna()).to_frame().rolling(ndays).apply(get_slope)
        frames.append(slope)   
    d1 = pd.concat(frames,axis=1)
    return d1
