#CASE STUDY STORM MOTORS
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('G:\\repository\\preownedcars\\')

data = pd.read_csv('cars_sampled.csv')
data1 = data.copy()

#SUMMARIZING DATA

print(data1.info())

summary = data1.describe()
pd.set_option('display.float_format', lambda x: '%.3f' % x)
data1.describe()
summary_cat = data1.describe(include = 'O')

pd.set_option('display.max_columns', 500)
data1.describe()

sns.set(rc= {'figure.figsize': (11.7,8.27)})

col = ['dateCrawled','name','dateCreated','postalCode', 'lastSeen']
data1 = data1.drop(columns = col, axis=1)

data1.drop_duplicates(keep='first',inplace = True)

#DATA CLEANING
#MISSING VALUES
print(data1.isnull().sum())
#data1['gearbox'].value_counts(dropna = False)
#print(data1['gearbox'].unique()) 
#print(data1['vehicleType'].unique())
#print(data1['model'].unique())

yearwise_count = data1['yearOfRegistration'].value_counts().sort_index()
sum(data1['yearOfRegistration']>2018)
sum(data1['yearOfRegistration']<1950)

sns.regplot(data= data1, x='yearOfRegistration', y='price',scatter =True, fit_reg = False)
#WORKING RANGE 1950-2018

#PRICE VARIABLE

price_count = data1['price'].value_counts().sort_index()
sum(data1['price']>150000)
sum(data1['price']<100)
data1['price'].describe() #skewness bwn mean and median
sns.boxplot(y=data1['price']) #outliers

#WORKING RANGE 100-150000

power_count  = data1['powerPS'].value_counts().sort_index()
sns.distplot(data1['powerPS'])
sns.boxplot(y=data1['powerPS'])
data1['powerPS'].describe()
sum(data1['powerPS']>500)
sum(data1['powerPS']<10)
sns.regplot(x='powerPS',y='price',scatter = True, fit_reg = False, data=data1)

#TRAIL AND ERROR 

#WORKING RANGE 10-500

