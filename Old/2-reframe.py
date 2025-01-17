#%% Find the countries that are at least four days ahead of Canada or have more cases
ndays = wrld.Canada.gt(90).sum()+4
list1 = list(wrld.max().index[wrld.gt(90).sum()>=ndays])
list2 = list(wrld.max().index[wrld.ge(wrld.Canada.max()).sum().gt(0)])
countries = list(set(list1+list2))
#countries.append('Canada')
countries.remove('World')
countries.remove('China')

wrld[countries].max().sort_values(ascending=False)

#%% Find the biggest provinces
can.gt(90).sum().sort_values(ascending=False)
provinces=['Alberta','Quebec','Ontario','British Columbia','Nova Scotia']
can[provinces].max().sort_values(ascending=False)

#%% Find the biggest states, the ones ahead of Alberta in days or cases
us.max().sort_values(ascending=False)
us.gt(90).sum().sort_values(ascending=False)

list1 = list(us.max().index[us.gt(can.Alberta.max()).sum().gt(0)])
list2 = list(us.max().index[us.gt(90).sum().ge(can.Alberta.gt(90).sum())])

#Ok, go with just days since cases is pretty much all of them
states = list2
us[states].max().sort_values(ascending=False)

#%% Reframe the dataframes as days since 100th case (actually 90th)
wrld = hundo(wrld,countries,90)
can = hundo(can,provinces,90)
us = hundo(us,states,90)


#%% Plot some stuff
wrld.plot(logy=True)
wrld.plot_bokeh.line(logy=True,figsize=(1200,800),xlim=(0,40))

us.plot_bokeh.line(logy=True,figsize=(1200,800))

can.plot_bokeh.line(logy=True,figsize=(1200,800))
