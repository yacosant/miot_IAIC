import numpy as np
import copy
import pandas as pd
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
import io
import seaborn as sns

def carga_csv(file_name):
    return read_csv(file_name, header=None).values

def cargar():
    return pd.read_csv('dataset.csv',
        nrows=10)   ###################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! max 10

def count(X):
    f = sns.countplot(x='target', data=X)
    f.set_title("Distribución de problemas de corazón")
    f.set_xticklabels(['No Problema de Corazón', 'Problema de Corazón'])
    plt.xlabel("")
    plt.show()

def countSex(X):
    sns.countplot(x='sex', data=df, palette="mako_r")
    plt.xlabel("Sex (0 = female, 1= male)")
    plt.show() 

############
#data = carga_csv('./dataset.csv')
#print(data)
data = cargar()
print(data.head(20))

count(data)
countSex(data)



