import numpy as np
import copy
import pandas as pd
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
import io
import seaborn as sns
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense



def carga_csv(file_name):
    return read_csv(file_name, header=None).values

def cargar(num=0):
    if num==0:
        return pd.read_csv('dataset.csv')
    else:
        return pd.read_csv('dataset.csv',nrows=num) 

def count(X):
    f = sns.countplot(x='target', data=X)
    f.set_title("Distribución de problemas de corazón")
    f.set_xticklabels(['Sin Problema de Corazón', 'Problema de Corazón'])
    plt.xlabel("")
    plt.savefig('./img/count.png')
    plt.show()

def countSex(X):
    f=sns.countplot(x='sex', data=X, palette="mako_r")
    f.set_title("Distribución de sexo de la muestra")
    f.set_xticklabels(['Mujeres', 'Hombres'])
    plt.savefig('./img/countSex.png')
    plt.show() 

def countSexProblemas(X):
    f = sns.countplot(x='target', data=X, hue='sex')
    plt.legend(['Mujer', 'Hombre'])
    f.set_title("Problemas de corazón por genero")
    f.set_xticklabels(['Sin Problema de Corazón', 'Problema de Corazón'])
    plt.xlabel("")
    plt.savefig('./img/countSexProblemas.png')
    plt.show()

def countByEdad(X):
    pd.crosstab(X.age,X.target).plot(kind="bar",figsize=(20,6))
    plt.title("Problemas de corazón por edad")
    plt.legend(['Sin Problema de Corazón', 'Problema de Corazón'])
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.savefig('./img/heartDiseaseAndAges.png')
    plt.show()

def pintar():
    count(data)
    countSex(data)
    countSexProblemas(data)
    countByEdad(data)

def keras(X_train, X_test, Y_train, Y_test):
    
    model = Sequential()
    model.add(Dense(30, input_dim=13, activation='tanh'))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=100, verbose=1)
    model.summary()
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    print('Model Accuracy = ',score[1])

############
#data = carga_csv('./dataset.csv')
#print(data)
data = cargar()
print(data.head(20))
#print(data.count())#303 casos
#pintar()

#print(data.loc[data['age'] == 29])

print("Media de edad: " + str(data['age'].mean()))

y = data.target.values
X_data = data.drop(['target'], axis = 1)
# Normalize
#X_data = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data)).values


X_train, X_test, Y_train, Y_test = train_test_split(X_data, y, test_size = 0.30, stratify= data['target'], random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print("---------")
#X_train=X_train.T
print(X_train.shape)
"""
print(Input_train['target'].mean())
print(Input_test['target'].mean())

OUT:
-sin stratify
0.5518867924528302
0.5274725274725275
-con stratify
0.5424528301886793
0.5494505494505495
"""

#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

keras(X_train, X_test, Y_train, Y_test)