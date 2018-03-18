# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 21:56:55 2018

@author: LMC
"""
# bibliotecas usadas
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


#lendo os conjuntos de teste 
train = pd.read_csv('train_titanic.csv')
test = pd.read_csv('test_titanic.csv')

#cinco primeira linhas do conjunto de treino 
print(train .head(5)) 

#retirando dados irrelevantes 
train .drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

#Criação de um DataFrame a partir do one-hot enconding
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

#quantidade de valores nulos no conjunto treino 
new_data_train .isnull().sum().sort_values(ascending=False).head(10)

#Preenchendo valores nulos 
new_data_train ['Age'].fillna(new_data_train ['Age'].mean(),inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(),inplace=True)

#Quantidade de valores nulos de teste 
new_data_test.isnull().sum().sort_values(ascending=False).head(10)

#Preenchendo os valores nulos 
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(),inplace=True)

#Separando features e target para criação do modelo
X = new_data_train .drop('Survived',axis=1)
y = new_data_train['Survived']
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

#Criação do modelo 
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
max_depth=3, min_samples_leaf=5)
clf_entropy.fit(train_X, train_y)

#verificando o Score do treino 


print(" \n Making predictions for the following 11 columns:\n")
print(X.head())
print("\nThe predictions are\n")
print(clf_entropy.predict(X.head()))


val_predictions = clf_entropy.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


#criando arquivo de subimmisão
submission = pd.DataFrame()
submission['PassengerId'] = new_data_test['PassengerId']
submission['Survived'] = clf_entropy.predict(new_data_test)
submission.to_csv('submission01.csv',index=False)



