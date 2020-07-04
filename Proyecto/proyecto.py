import numpy as np
import helper as help
import copy
import pandas as pd
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
import io
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from termcolor import colored
from keras.optimizers import Adam#########
from keras.models import Sequential
from keras.layers import Dense

options = ["[0] - Salir", "[1] - Generar Gráficas", "[2] - Mostrar información de la muestra", "[3] - Mostrar elementos de la muestra","[4] - Ejecución:[Sklearn] Regresión Logistica", "[5] - Ejecución: [Keras-basic]"]
nombres =["Sklearns-RegresionLogistica", "Keras-basic"]
ejecuciones ={}
historicos ={}

def carga_csv(file_name):
    return read_csv(file_name, header=None).values

def cargar(num=0):
    if num==0:
        return pd.read_csv('dataset.csv')
    else:
        return pd.read_csv('dataset.csv',nrows=num) 

def menu():
    print(colored("==============================================", 'green'))
    print(colored("MENÚ:", 'green'))
    #for op in options:
    i = 1
    while i < len(options):
        print(colored("\t"+options[i], 'green'))
        i+=1

    print(colored("\t"+options[0], 'green'))

def cabecera(op):
    if op < int(len(options)):
        num = len(options[op])
        half = "=" * int((47 - num)/2) 
        print(colored(half+options[op]+half, "red"))


def count(X):
    f = sns.countplot(x='target', data=X)
    f.set_title("Distribución de problemas de corazón")
    f.set_xticklabels(['Sin Problema de Corazón', 'Problema de Corazón'])
    plt.xlabel("")
    plt.savefig('count.png')
    plt.show()

def countSex(X):
    f=sns.countplot(x='sex', data=X, palette="mako_r")
    f.set_title("Distribución de sexo de la muestra")
    f.set_xticklabels(['Mujeres', 'Hombres'])
    plt.savefig('countSex.png')
    plt.show() 

def countSexProblemas(X):
    f = sns.countplot(x='target', data=X, hue='sex')
    plt.legend(['Mujer', 'Hombre'])
    f.set_title("Problemas de corazón por genero")
    f.set_xticklabels(['Sin Problema de Corazón', 'Problema de Corazón'])
    plt.xlabel("")
    plt.savefig('countSexProblemas.png')
    plt.show()

def countByEdad(X):
    pd.crosstab(X.age,X.target).plot(kind="bar",figsize=(20,6))
    plt.title("Problemas de corazón por edad")
    plt.legend(['Sin Problema de Corazón', 'Problema de Corazón'])
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.savefig('heartDiseaseAndAges.png')
    plt.show()

def pintar(data):
    count(data)
    countSex(data)
    countSexProblemas(data)
    countByEdad(data)

def prepararData(data):
    y = data.target.values
    X_data = data.drop(['target'], axis = 1)
    # Normalize
    X_data = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data)).values


    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y, test_size = 0.30, stratify= data['target'], random_state = 5)
    """
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    print("---------")
    #X_train=X_train.T
    print(X_train.shape)
    """
    return X_train, X_test, Y_train, Y_test

def compararAciertos():
    print(ejecuciones) #debug
    sns.set_style("whitegrid")
    plt.figure(figsize=(16,5))
    plt.yticks(np.arange(0,100,10))
    plt.ylabel("% de Presición")
    plt.xlabel("Ejecuciones")
    sns.barplot(x=list(ejecuciones.keys()), y=list(ejecuciones.values()))
    plt.show()

def sklearnLogisticRegression(X_train, X_test, Y_train, Y_test, graf = False):
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train,Y_train)
    acc = model.score(X_test,Y_test)*100

    ejecuciones['Sklearns-RegresionLogistica'] = acc
    print(colored("Precisión: " +str(acc)+" %", "blue"))

def keras(X_train, X_test, Y_train, Y_test, model = 0, graf = True, name='Keras-basic'):
    if model == 0:
        model = Sequential()
        model.add(Dense(30, input_dim=13, activation='tanh'))
        model.add(Dense(20, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    
    #history = model.fit(X_train, Y_train, epochs=200, verbose=2)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, verbose=2)
    historicos[name]= history
    #model.summary()
    score = model.evaluate(X_test, Y_test, verbose=0)
    acc=score[1]*100
    ejecuciones['Keras-basic'] = acc
    #print(colored("Presición entrenamiento: " +str(history.history['acc'])+" %", "blue"))
    print(colored("Presición test: " +str(acc)+" %", "blue"))
    help.plot_loss_accuracy(history)
    plt.savefig('history-'+name+'.png')
    if graf == True:
        plt.show()
    help.plot_confusion_matrix(model, X_train, Y_train)
    plt.savefig('confusion_matrix-'+name+'.png')
    if graf == True:
        plt.show()
    
        
    

def main():
    os.environ['KMP_WARNINGS'] = '0' #Desactiva logs de INFO de keras si se pone a 0 cuando no se usan
    data = cargar()
    X_train, X_test, Y_train, Y_test= prepararData(data)
    
    while True:
        menu()
        op = int(input("Selecciona una opción >> "))
        cabecera(op)

        if op==1:
            print("Pintando y Guardando gráficas...")
            pintar(data)

        elif op==2:
            print(data.describe())
            

        elif op==3:
            num= int(input("¿Cuantos elementos quieres recuperar? (1-303): "))
            if num <303 and num>1:
                print(data.head(num))
            else: print(colored("El número no es correcto", "red"))

        elif op==4: #Sklearn
            print("")
            sklearnLogisticRegression(X_train, X_test, Y_train, Y_test)
        
        elif op==5: #Keras Basic
            print(colored("Keras con 3 capas densas:", "red"))
            print(colored("Nodos: [30-tanh][20-tanh][1-sigmoid]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                keras(X_train, X_test, Y_train, Y_test, 0, True)
            else: print("No ejecutado")

        elif op==6: #Keras 
            print(colored("Keras con 4 capas densas:", "red"))
            print(colored("Nodos: [30-tanh][20-tanh][1-sigmoid][1-sigmoid]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                model = Sequential()
                model.add(Dense(30, input_dim=13, activation='sigmoid'))
                model.add(Dense(20, activation='tanh'))
                model.add(Dense(10, activation='sigmoid'))
                model.add(Dense(1, activation='tanh'))
                #.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

                keras(X_train, X_test, Y_train, Y_test, model, True,"k6")
            else: print("No ejecutado")
        
        elif op==7: #keras
            print("")
            model = Sequential()
            """"
            model.add(Dense(200, input_dim=13, activation='relu'))
            model.add(Dense(150, activation='relu'))
            model.add(Dense(100, activation='relu'))
            model.add(Dense(50, activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
            """
            """
            print("(1) - [4-tanh]-[2-tanh]-[10-softmax]")
            print("(2) - [64-tanh]-[32-tanh]-[16-softmax]-[10-softmax]")
            print("(3) - [64-tanh]-[32-tanh]-[16-softmax]-[10-sigmoid]")
            print("(4) - [64-tanh]-[32-tanh]-[16-tanh]-[10-sigmoid]")
            print("(5) - [64-tanh]-[32-tanh]-[16-tanh]-[10-softmax]")
            print("(6) - [128-tanh]-[64-tanh]-[32-tanh]-[10-softmax]")
            """
            """
            model.add(Dense(4, input_dim=13, activation='tanh'))
            model.add(Dense(2, activation='tanh'))
            model.add(Dense(1, activation='softmax'))
            54.94505763053894 %
            """
            """
            model.add(Dense(64, input_dim=13, activation='tanh'))
            model.add(Dense(32, activation='tanh'))
            model.add(Dense(16, activation='softmax'))
            model.add(Dense(1, activation='softmax'))
            Presición: 54.94505763053894 %
            """
            """
            model.add(Dense(64, input_dim=13, activation='tanh'))
            model.add(Dense(32, activation='tanh'))
            model.add(Dense(16, activation='softmax'))
            model.add(Dense(1, activation='sigmoid'))
            Presición: 76.92307829856873 %
            """
            """
            model.add(Dense(64, input_dim=13, activation='tanh'))
            model.add(Dense(32, activation='tanh'))
            model.add(Dense(16, activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
            Presición: 74.72527623176575 %
            """

            """
            model.add(Dense(64, input_dim=13, activation='tanh'))
            model.add(Dense(32, activation='tanh'))
            model.add(Dense(16, activation='tanh'))
            model.add(Dense(1, activation='softmax'))
            Presición: 54.94505763053894 %
            """

            model = Sequential()
            model.add(Dense(300, input_dim=13, activation='tanh'))
            model.add(Dense(200, activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

            keras(X_train, X_test, Y_train, Y_test, model, True,"k7")

        elif op==8: #Ejecutar todos 
            print("")
            model = Sequential()
            model.add(Dense(50, input_dim=13, activation='sigmoid'))
            #model.add(Dense(100, activation='tanh'))
            model.add(Dense(10, activation='sigmoid'))
            model.add(Dense(200, activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

            keras(X_train, X_test, Y_train, Y_test, model, True,"k8")

        elif op==9: #Comparar graficas 
            print("")

        elif op==0:
            print("Saliendo...")
            break
        else:
            print (colored("La opción elegida no es correcta...\n","red"))



main()  
############
#data = carga_csv('./dataset.csv')
#print(data)
data = cargar()
#print(data.head(20))
#print(data.count())#303 casos
#pintar()

#print(data.loc[data['age'] == 29])

#print("Media de edad: " + str(data['age'].mean()))

X_train, X_test, Y_train, Y_test= prepararData(data)

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

#sklearnLogisticRegression(X_train, X_test, Y_train, Y_test)

ejecuciones['Prueba']=95

keras(X_train, X_test, Y_train, Y_test)
compararAciertos()


"""
    Model Accuracy =  0.7362637519836426 SIN NORMALIZAR
    Model Accuracy =  0.8351648449897766 NORMALIZADO
"""


"""
class Producto:
    def __init__(self, nombre, precio):
        self.nombre = nombre
        self.precio = precio

"""