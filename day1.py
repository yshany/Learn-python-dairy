
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
os.chdir('E:/ti')
data=pd.read_csv('train.csv')
data.head()
data.info()
data2=data.drop(['PassengerId','Name','Ticket'],axis=1)
dtest=pd.read_csv('test.csv')
dtest2=data.drop(['Name','Ticket'],axis=1)
pd.Series(dtest.columns)
dt=np.array(dtest)
dt[:,range(0,9)]
data['Fare'].groupby([data['Survived'],data['Sex']]).mean().unstack()
data['Survived'].groupby(data['Sex']).value_counts().unstack()
pd.get_dummies(dtest)
data['Survived','Embarked']
dp=data.get(['Sex','Fare','Age'])
sum(data['Survived'][dp['Age']!=dp['Age']])/891
sum(data['Survived'][dp['Age']>=40])/891
sum(data['Survived'][dp['Fare']!=dp['Fare']])/891
sum(data['Survived'])/891
len(data['Survived'][dp['Fare']!=dp['Fare']])
len(dtest[np.isnan(dtest['Age'])])
dtest['Fare']=dtest['Fare'].fillna(data['Fare'].mean())
from sklearn import svm
svc=svm.SVC(C=1000,degree=1)
svc.fit(np.array(pd.get_dummies(data.get(['Sex','Fare'])))[:,range(0,2)],np.array((data['Survived'])))
sum(svc.predict(np.array(pd.get_dummies(dtest.get(['Sex','Fare'])))[:,range(0,2)]))/418
sub=pd.DataFrame({'PassengerId':dtest['PassengerId'],'Survived':svc.predict(np.array(pd.get_dummies(dtest.get(['Sex','Fare'])))[:,range(0,2)])})
sub.to_csv('first.csv',index=False)
