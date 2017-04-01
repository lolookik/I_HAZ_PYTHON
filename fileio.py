'''read dataframe from file to list'''
#required for the graph to appear
import matplotlib
matplotlib.use('Qt4Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#test matplotlib
x = np.arange(0, 5, 0.1);
y = np.sin(x)
plt.plot(x,y)


#read file as UTF-16LE into DataFrame
df = pd.read_csv('95-2017_03_13-19.txt', encoding='UTF-16LE', 
							sep='\t', 
							parse_dates=['ДатаВремя'], 
							dayfirst=True)


#check beginning
print(df.head())
#extract collumn
dat = df['ДатаВремя']
best_demand = df['Лучший cпрос']
#print(best_demand.head())
print(dat.head())
print(best_demand.head())
plt.plot(dat, best_demand)
plt.show()
