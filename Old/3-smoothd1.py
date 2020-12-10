#%% Calculate the five-day smoothed first derivative
wrldd1 = getd1(wrld,5)
cand1 = getd1(can,5)
usd1 = getd1(us,5)

#%% Plot a few
wrldd1.plot()
wrldd1.plot_bokeh(figsize=(1200,800))
cand1.plot_bokeh(figsize=(1200,800))
usd1.plot_bokeh(figsize=(1200,800))

#%% Combine the cases and growth rate in a single dataframe
comb = pd.concat([wrld.stack().to_frame(),wrldd1.stack().to_frame()],axis=1)
comb.dropna(inplace=True)
comb.reset_index(level=1,inplace=True)
comb.columns = ['Country','C','d1']
comb.plot_bokeh.scatter(x='C',y='d1',category='Country',logx=True,figsize=(1200,800))

#%% Do this for Canada
comb = pd.concat([can.stack().to_frame(),cand1.stack().to_frame()],axis=1)
comb.dropna(inplace=True)
comb.reset_index(level=1,inplace=True)
comb.columns = ['Province','C','d1']
comb.plot_bokeh.scatter(x='C',y='d1',category='Province',logx=True,figsize=(1200,800))


#%% The US
comb = pd.concat([us.stack().to_frame(),usd1.stack().to_frame()],axis=1)
comb.dropna(inplace=True)
comb.reset_index(level=1,inplace=True)
comb.columns = ['State','C','d1']
comb.plot_bokeh.scatter(x='C',y='d1',category='State',logx=True,figsize=(1200,800))

