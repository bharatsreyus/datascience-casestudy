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


#WORKING RANGE OF DATA

data1 = data1[
    (data1.yearOfRegistration<=2018)
  & (data1.yearOfRegistration>=1950)
  & (data1.price>=100)
  & (data1.price<=150000)
  & (data1.powerPS>=10)
  & (data1.powerPS<=500)]

data1['monthOfRegistration']/=12
data1['Age'] = (2018 - data1['yearOfRegistration'] + data1['monthOfRegistration'])
data1['Age'] = round(data1['Age'],2)
data1['Age'].describe()

#REMOVE TWO COLUMNS - monthofreg and yearofreg
data1 = data1.drop(['yearOfRegistration', 'monthOfRegistration'],axis = 1)

#VISUALISING PARAMETERS

sns.distplot(data1['Age'])
sns.boxplot(y=data1['Age'])

sns.distplot(data1['price'])
sns.boxplot(y=data1['price'])

sns.distplot(data1['powerPS'])
sns.boxplot(y=data1['powerPS'])

sns.regplot(x= data1['Age'], y=data1['price'],fit_reg = False, data=data1)

sns.regplot(x= data1['powerPS'], y=data1['price'],fit_reg = False, data=data1) 

#FREQUENCIES OF PARAMTERS
data1['seller'].value_counts()
pd.crosstab(data1['seller'],columns = 'count', normalize=True)
#commercial IS INSIGNIFICANT

data1['offerType'].value_counts()
#ALL ARE offer => INSIGNIFICANT

data1['abtest'].value_counts()
pd.crosstab(data1['abtest'], columns = 'count', normalize =True)
sns.countplot(x= 'abtest',data = data1)

#EQUALLY DISTRIBUTED

sns.boxplot(x='abtest', y='price',data = data1)

#50-50 DISTRIBUTION FOR EVERY PRICE VALUE
#ABTEST => INSIGNIFICANT 

data1['vehicleType'].value_counts()
pd.crosstab(data1['vehicleType'],columns = 'count',normalize =True)
sns.countplot(x=data1['vehicleType'])
sns.boxplot(x='vehicleType',y='price',data=data1)
#VEHICLE TYPE AFFECTS PRICE HENCE RETAIN IT

#VARIABLE GEARBOX

data1['gearbox'].value_counts()
pd.crosstab(data1['gearbox'],columns='count',normalize=True)
sns.countplot(data1['gearbox'])
sns.boxplot(data1['gearbox'],data1['price'])
#GEARBOX AFFECTS PRICE

data1.columns

#VARIABLE MODEL

data1['model'].value_counts()
pd.crosstab(data1['model'],columns='count',normalize=True)
sns.countplot(data1['model'])
#HIGLY DISTRIBUTED SO CONSIDERED

#VARIABLE KILOMETER

data1['kilometer'].value_counts()
pd.crosstab(data1['kilometer'], columns='counts',normalize=True)
sns.countplot(data1['kilometer'])
data1['kilometer'].describe()
#CONSIDERED

#VARIABLE FUELTYPE

data1['fuelType'].value_counts().sort_index()
data1['fuelType'].describe()
sns.boxplot(x='fuelType',y='price',data=data1)
pd.crosstab(data1['fuelType'],columns='count',normalize=True)
#CONSIDERED

data1.columns

#VARIABLE BRAND

data1['brand'].value_counts()
pd.crosstab(data1['brand'],columns='counts',normalize=True)
sns.countplot(data1['brand'])
#CONSIDERED

#VARIABLE NOTREPAIREDDAMAGED

data1['notRepairedDamage'].value_counts()
pd.crosstab(data1['notRepairedDamage'],columns='counts',normalize=True)
sns.boxplot(x='notRepairedDamage',y='price',data=data1)
sns.countplot(data1['notRepairedDamage'])
data1['notRepairedDamage'].describe()
#FALL UNDER LOWER PRICE RANGE

#REMOVING INSIGNIFICANT VARIABLES
col = ['seller','abtest','offerType']
data1.drop(columns=col,axis=1,inplace=True)
data1_copy = data1.copy()

#CORRELATIONS

cars_select1 = data1.select_dtypes(exclude=[object])
correlation = cars_select1.corr()
round(correlation,3)
correlation.loc[:,'price'].abs().sort_values(ascending=False)[1:]

#TWO MODELS: 1.LINEAR REGRESSION 2.RANDOM FOREST
#BY OMITTING MISSING VALUE ROWS
#BY NOT OMITTING

#OMITTING MISSING VALUES

data1_omit = data1.dropna(axis=0)

#CONVERTING CATEGORICAL TO DUMMY VARIABLES

data1_omit = pd.get_dummies(data1_omit, drop_first=True)

#IMPORT NECESSARY LIBRARIES

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#BUILDING MODEL WITH OMITTED DATA

x1 = data1_omit.drop(['price'],axis='columns',inplace=False)
y1 = data1_omit['price']

prices = pd.DataFrame({'1.before':y1,'2.after':np.log(y1)})
prices.hist()

y1=np.log(y1)


#TRAIN TEST SPLIT

X_train,X_test,y_train,y_test = train_test_split(x1,y1, test_size = 0.3, random_state=3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#SET BASE MODEL TO COMPARE WITH REGRESSION MODEL

#MEAN VALUE: BENCH MARK

base_pred = np.mean(y_test)
print(base_pred)

base_pred = np.repeat(base_pred, len(y_test))

base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print(base_root_mean_square_error)


#LINEAR REGRESSION MODEL

lgr = LinearRegression(fit_intercept=True)
model_lin1 = lgr.fit(X_train,y_train)
prediction_lin1 = model_lin1.predict(X_test)

#FINDING PERFORMANCE USING MSE AND RMSE
lin_mse1 = mean_squared_error(y_test, prediction_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_mse1,lin_rmse1)

#R SQUARED VALUE

r_test1 = model_lin1.score(X_test, y_test)
r_train1 = model_lin1.score(X_train,y_train)
print(r_test1,r_train1)

#REGRESSION DIAGNOSTICS: RESIDUAL ANALYSIS

residual1 = y_test - prediction_lin1
sns.regplot(x=prediction_lin1,y=residual1,scatter=True,fit_reg=False)
residual1.describe()


#RANDOM FORREST WITH OMITTED DATA

rf = RandomForestRegressor(n_estimators = 100, max_features= 'auto',
                           max_depth = 100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)

#MODEL
model_rf1 = rf.fit(X_train,y_train)

#PREDICT

prediction_rf1= rf.predict(X_test)

#MSE RMSE

rf_mse1 = mean_squared_error(y_test, prediction_rf1)
rf_rsme1 = np.sqrt(rf_mse1)
print(rf_rsme1)

#R SQUARED VALUES

r_rf_test1 = model_rf1.score(X_test,y_test)
r_rf_train1 = model_rf1.score(X_train,y_train)
print(r_rf_test1,r_rf_train1)

#MODEL BUILDING WITH IMPUTED DATA



