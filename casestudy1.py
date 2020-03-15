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
