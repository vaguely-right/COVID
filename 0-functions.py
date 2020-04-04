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
def get_slope(Y):
    n = len(Y)
    X = range(0,n)
    p = np.polyfit(X,Y,1)
    return p[0]

def make_pred(df):
    Y = df.dropna().to_numpy()
    X = df.dropna().index.values
    n = X.max()+1
    p = np.polyfit(X,Y,1)
    pred = np.polyval(p,n)
    return pred

def rsq(x,y):
    r = np.corrcoef(x,y)[1,0]**2
    return r

def hundo(df,cols):
    frames = []
    for c in cols:
        frames.append(df[df[c]>=90].reset_index()[c].to_frame())
    h = pd.concat(frames,axis=1)
    h = h[h.max().sort_values(ascending=False).index]
    return h

