# terror
predictions about attacks that could happen if any attack take place in different states of countries
import pandas as pd
from pandas import *
import numpy as np
from numpy import *
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score as AUC

data=pd.read_csv('terror.csv')
#data=data['provstate'].fillna('unknown',inplace=True)
#data1=data[['country','provstate','attacktype1','targtype1','weaptype1']]
data1=data[data['country_txt']=='India']
data1.provstate.fillna('unknown')
data1=data1[data1['provstate']!='unknown']
attack=data1.ix[:,['attacktype1_txt',]]
training_set_for_attacks=data1.ix[:,['provstate','targtype1_txt','weaptype1_txt']]
training_set_for_weapons=data1.ix[:,['provstate','targtype1_txt','attacktype1_txt']]
trainning_set_for_targets=data1.ix[:,['provstate','weaptype1_txt','attacktype1_txt']]
target=data1.ix[:,['targtype1_txt']]
weaptype=data1.ix[:,['weaptype1_txt']]
            

from sklearn.preprocessing import LabelEncoder
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,data1,y=None):
        return self

    def transform(self,data1):
       
        output = data1.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,data1,y=None):
        return self.fit(data1,y).transform(data1)

y_attacks_train=MultiColumnLabelEncoder().fit_transform(attack)

x_attacks_train=MultiColumnLabelEncoder().fit_transform(training_set_for_attacks)
y_weaptype_train=MultiColumnLabelEncoder().fit_transform(weaptype)
y_target_train=MultiColumnLabelEncoder().fit_transform(target)
x_target_train=MultiColumnLabelEncoder().fit_transform(trainning_set_for_targets)
x_weaptype_train=MultiColumnLabelEncoder().fit_transform(training_set_for_weapons)



x_atacks_train, x_attacks_test, y_attack_train, y_attack_test = train_test_split(x_attacks_train,y_attacks_train,test_size=0.2)
rf1= RandomForestClassifier(random_state=123)

predict=rf1.fit(x_atacks_train,y_attack_train.values.ravel()).predict(x_attacks_test)
plt.figure(2, figsize=(8, 6))
scores= rf1.score(x_attacks_test, y_attack_test)
plt.plot(predict)
print 'Accuracy:', AUC(y_attack_test, predict)
rf2= RandomForestClassifier(random_state=123)
x_weaptype_train, x_weaptype_test, y_weaptype_train, y_weaptype_test = train_test_split(x_weaptype_train,y_weaptype_train,test_size=0.2)
predict1=rf2.fit(x_weaptype_train,y_weaptype_train.values.ravel()).predict(x_weaptype_test)
print 'Accuracy:', AUC(y_weaptype_test, predict1)
rf3= RandomForestClassifier(random_state=123)
x_target_train, x_target_test, y_target_train, y_target_test = train_test_split(x_target_train,y_target_train,test_size=0.2)
predict2=rf3.fit(x_target_train,y_target_train.values.ravel()).predict(x_target_test)
print 'Accuracy:', AUC(y_target_test, predict2)

d=[predict,predict1,predict2]
c=[[],[],[]]
for i in range(len(d)):
    for j in d[i]:
        c[i].append(j)


futureattacks=pd.DataFrame({'provstate':data1['provstate'][7952:],'attacktype1_txt':data1['attacktype1_txt'][7952:],'weaptype1_txt':data1['weaptype1_txt'][7952:],'targtype1_txt':data1['targtype1_txt'][7952:],'predicted_attacks':c[0],'predicted_weaptype':c[1],'predicted_target':c[2]})
attacks_type=futureattacks.groupby(['provstate','attacktype1_txt'])['predicted_attacks'].mean()
weaptype_type=futureattacks.groupby(['provstate','weaptype1_txt'])['predicted_weaptype'].mean()
target_type=futureattacks.groupby(['provstate','targtype1_txt'])['predicted_target'].mean()

attacks_type.unstack().plot(kind='barh', stacked=True)
weaptype_type.unstack().plot(kind='barh', stacked=True)
target_type.unstack().plot(kind='barh', stacked=True)

