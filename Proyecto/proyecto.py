import numpy as np
import copy
import pandas as pd
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
import io
import seaborn as sns
from sklearn.model_selection import train_test_split

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


############
#data = carga_csv('./dataset.csv')
#print(data)
data = cargar()
print(data.head(20))
#print(data.count())#303 casos
#pintar()

#print(data.loc[data['age'] == 29])

print("Media de edad: " + str(data['age'].mean()))

Input_train, Input_test, Target_train, Target_test = train_test_split(data, data['target'], test_size = 0.30, stratify= data['target'], random_state = 5)
print(Input_train.shape)
print(Input_test.shape)
print(Target_train.shape)
print(Target_test.shape)
print("---------")

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