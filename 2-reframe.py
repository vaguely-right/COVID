#%% Find the countries that are at least four days ahead of Canada
ndays = wrld.Canada.gt(90).sum()+4
countries = list(wrld.max().index[wrld.gt(90).sum()>=ndays])
countries.append('Canada')
countries.remove('World')

wrld[countries].max().sort_values(ascending=False)

#%% Find the biggest provinces
can.gt(90).sum().sort_values(ascending=False)
provinces=['Alberta','Quebec','Ontario','British Columbia']
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


