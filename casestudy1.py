#CASE STUDY SUBSIDY.INC
import pandas as pd
import numpy as np
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

os.chdir('G:\\repository\\income\\')

data_income = pd.read_csv('income.csv')
data = data_income.copy()

#EXPLORATORY DATA ANALYSIS
#1.GETTING TO KNOW THE DATA
print(data.info())

#2.DATA PROCESSING - MISSING VALUES
print('Missing values info:\n', data.isnull().sum())

summary_data = data.describe()
print('Description of data:\n', summary_data)
summary_cate = data.describe(include = 'O')
print(summary_cate)

data['JobType'].value_counts()
data['occupation'].value_counts()

print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
#THERE EXISTS ' ?' IN PLACE OF NAN

data = pd.read_csv('income.csv', na_values = [' ?'])

#DATA PRE PROCESSING
data.isnull().sum()

missing  = data[data.isnull().any(axis = 1)]

'''POINTS TO NOTE
1. JOBTYPE MISSIN VALUES : 1809
2. OCCUPATION MISSIN VALUES : 1816
3. 1809 ROWS HAVE BOTH JOBTYPE AND COLUMNS MISSIN 
4. 7 ROWS HAVE OCCUPATION AS MISSIN COZ JOBTYPE IS 'NEVERWORKED'
'''

data2 = data.dropna(axis = 0)

#VISUALIZATION

correlation = data2.corr()

data2.columns

gender = pd.crosstab(index = data2['gender'], columns = 'counts' , normalize = True)
print(gender)
gender_salstat = pd.crosstab(index = data2['gender'], columns = data2['SalStat'], margins = True, normalize = 'index')
print(gender_salstat)

sns.countplot(data2['SalStat'])

sns.distplot(data2['age'], bins =10, kde = False)

sns.boxplot('SalStat', 'age', data = data2)
data2.groupby('SalStat')['age'].median()

sns.pairplot(data2,hue = 'SalStat')

#LOGISTIC REGRESSION

data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
print(data2['SalStat'])
#data2['SalStat'] = data2['SalStat'].notnull().astype('int64') #mytoouch :)

new_data = pd.get_dummies(data2,drop_first = True)

columns_list = list(new_data.columns)
print(columns_list)

features = list(set(columns_list) - set(['SalStat']))
print(features)
 
y= new_data['SalStat'].values
print(y)
x=new_data[features].values
print(x)

#SPLITTING THE DATA
train_x,test_x,train_y,test_y = train_test_split(x,y, test_size = 0.3, random_state=0)

#INSTANCE OF THE MODEL
logistic = LogisticRegression()

logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#PREDICTION
prediction = logistic.predict(test_x)
print(prediction)

#CONFUSION MATRIX
confusion_matrix = confusion_matrix(test_y,prediction)
print(confusion_matrix)
    
accuracy = accuracy_score(test_y, prediction)
print(accuracy)

#IMPROVE THE ACCURACY BY REMOVING INSIGNIFICANT DATA

data2['SalStat'] = data2['SalStat'].map({' less than or equal to 50,000': 0, ' greater than 50,000': 1})
print(data2['SalStat'])

cols = ['gender', 'nativecountry', 'race', 'JobType']
new_data = data2.drop(cols,axis = 1)
new_data = pd.get_dummies(new_data , drop_first = True )
columns_list = new_data.columns
print(columns_list)

features = list(set(columns_list) - set(['SalStat']))
print(features)

#OUTPUT VALUES IN y

y=new_data['SalStat'].values
print(y)

#INPUT VALUES IN x
x=new_data[features].values
print(x)

train_x,test_x,train_y,test_y = train_test_split(x,y, test_size = 0.3, random_state=0)

logistic = LogisticRegression()

logistic.fit(train_x, train_y)

prediction = logistic.predict(test_x)

accuracy = accuracy_score(test_y, prediction)
print(accuracy)


#KNN

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

KNN_classifier = KNeighborsClassifier(n_neighbors = 10)

KNN_classifier.fit(train_x,train_y)

prediction = KNN_classifier.predict(test_x)

'''c_matrix = confusion_matrix(test_y, prediction)
print(c_matrix)'''

acc = accuracy_score(test_y,prediction)
print(acc)
