import numpy as np
#import helper as help
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
from sklearn.metrics import confusion_matrix
from termcolor import colored
from keras.models import Sequential
from keras.layers import Dense

options = ["[0] - Salir", "[1] - Generar Gráficas de la muestra", "[2] - Mostrar información de la muestra", "[3] - Mostrar elementos de la muestra","[4] - Ejecución:[Manual]  Regresión Logistica" ,"[5] - Ejecución:[Sklearn] Regresión Logistica", "[6] - Ejecución:[Keras1]  [64-tanh][32-tanh][16-tanh][1-softmax]", "[7] - Ejecución:[Keras2]  [4-tanh][2-tanh][1-sigmoid]", "[8] - Ejecución:[Keras3]  [30-tanh][20-tanh][1-sigmoid]", "[9] - Ejecución:[Keras4]  [50-sigmoid][10-sigmoid][200-tanh][1-sigmoid] - ¡MEJOR!", "[10]- Ejecución:[TODAS]   [Graficas de Precisión y de Confusión] - Comparar"]
ejecuciones ={}
confusiones ={}
historicos ={}
nombres = []

###################################################
#Funciones sacadas del archivo helper.py proporcionado por el profesor

def plot_loss_accuracy(history):
    """
    Genera una figura con la evolución del coste y la Precisión durante
    el entrenamiento de un modelo en Keras. 

    Argumentos:
    history -- un objeto History devuelto por el método fit de Keras
    https://keras.io/models/model/#fit
    """

    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['acc'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    plt.close(1)

def plot_confusion_matrix(model, X, y):
    """
    Genera una figura con un mapa de calor que representa la matriz
    de confusión del modelo 'model' aplicado sobre los datos X comparado
    con las etiquetas de y

    Argumentos:
    model -- modelo de Keras
    X, y -- datos y etiquetas
    """

    y_pred = model.predict_classes(X, verbose=0)
    plt.figure(figsize=(8, 6))
    conf =pd.DataFrame(confusion_matrix(y, y_pred))
    sns.heatmap(conf, annot=True, fmt='d',
                cmap='YlGnBu', alpha=0.8, vmin=0)
    return conf
###################################################

def cargar(num=0):
    if num==0:
        return pd.read_csv('dataset.csv')
    else:
        return pd.read_csv('dataset.csv',nrows=num) 

def prepararData(data):
    y = data.target.values
    X_data = data.drop(['target'], axis = 1)
    # Normalize
    X_data = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data)).values

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y, test_size = 0.30, stratify= data['target'], random_state = 5)
    return X_train, X_test, Y_train, Y_test

def menu():
    print(colored("==============================================", 'green'))
    print(colored("MENÚ:", 'green'))
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

def compararAciertos():
    print(ejecuciones) #debug
    sns.set_style("whitegrid")
    plt.figure(figsize=(16,5))
    plt.yticks(np.arange(0,100,10))
    plt.ylabel("% de Presición")
    plt.xlabel("Ejecuciones")
    sns.barplot(x=list(ejecuciones.keys()), y=list(ejecuciones.values()))
    plt.show()
    plt.savefig('Compararcion-Aciertos.png')

###################Regresion logistica manual

def initialize(dimension):
    return np.full((dimension,1),0.01)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forwardBackward(weight,x_train,y_train):
    # Forward
    y_head = sigmoid(np.dot(weight.T,x_train))
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    # Backward
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    return cost,derivative_weight


def update(weight,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    
    #for each iteration, update weight values
    for i in range(iteration):
        cost,derivative_weight = forwardBackward(weight,x_train,y_train)
        weight = weight - learningRate * derivative_weight
        
        costList.append(cost)
        index.append(i)

    print("iteration:",iteration)
    print("cost:",cost)
    return weight


def predict(weight,x_test):
    z = np.dot(weight.T,x_test) 
    y_head = sigmoid(z)

    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction


def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration, graf = True):
    dimension = x_train.shape[0]
    weight= initialize(dimension)
    weightNuevo = update(weight,x_train,y_train,learningRate,iteration)

    y_prediction = predict(weightNuevo,x_test)
    conf =pd.DataFrame(confusion_matrix(y_test, y_prediction.T))
    confusiones['RegresiónLogica-Manual'] = conf
    if graf == True:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf, annot=True, fmt='d',cmap='YlGnBu', alpha=0.8, vmin=0)
        plt.show()

    acc =(100 - np.mean(np.abs(y_prediction - y_test))*100)
    ejecuciones['RegresiónLogica-Manual'] = acc
    nombres.append('RegresiónLogica-Manual')
    print(colored("[RegresiónLogica-Manual]Precisión Test: " +str(acc)+" %", "blue"))

########################################################

def sklearnLogisticRegression(X_train, X_test, Y_train, Y_test, graf = True):
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train,Y_train)
    acc = model.score(X_test,Y_test)*100
    ejecuciones['Sklearns-RegresionLogistica'] = acc
    nombres.append('Sklearns-RegresionLogistica')

    y_pred = model.predict(X_test)
    conf =pd.DataFrame(confusion_matrix(Y_test, y_pred))
    confusiones['Sklearns-RegresionLogistica'] = conf
    print(colored("[Sklearns-RegresionLogistica]Precisión Test: " +str(acc)+" %", "blue"))
    if graf == True:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf, annot=True, fmt='d',cmap='YlGnBu', alpha=0.8, vmin=0)
        plt.savefig('confusion_matrix-Sklearns-RegresionLogistica.png')
        plt.show()
    

def keras(X_train, X_test, Y_train, Y_test, model = 0, graf = True, name='Keras-basic'):
    if model == 0:
        model = Sequential()
        model.add(Dense(30, input_dim=13, activation='tanh'))
        model.add(Dense(20, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, verbose=2)
    historicos[name]= history
    nombres.append(name)
    score = model.evaluate(X_test, Y_test, verbose=0)
    acc=score[1]*100
    ejecuciones[name] = acc
    print(colored("["+name+"] Presición test: " +str(acc)+" %", "blue"))
    if graf == True:
        plot_loss_accuracy(history)
        plt.savefig('history-'+name+'.png')
        plt.show()
    confusiones[name] = plot_confusion_matrix(model, X_train, Y_train)
    plt.savefig('confusion_matrix-'+name+'.png')
    if graf == True:
        plt.show()
    else: 
        plt.close()
    
def compararModelosConfusion():
    plt.figure(figsize=(24,12))
    plt.suptitle("Comparación Matrices de Confusión",fontsize=24)
    plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

    tam=len(confusiones)
    for i in range(tam):
        plt.subplot(2,3,i+1)
        plt.title(nombres[i])
        #plt.title(i)
        sns.heatmap(confusiones[nombres[i]],annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
    plt.show()
    plt.savefig('Compararcion-Confusion.png')


def compararModelosHistoricos():
    plt.figure(figsize=(24,12))
    plt.suptitle("Comparación Historicos",fontsize=24)
    plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

    tam=len(nombres)
    i = 0
    j = 0
    while i < tam:
        if nombres[i]!='Sklearns-RegresionLogistica' and nombres[i]!='RegresiónLogica-Manual':
            j+=1
            plt.subplot(2,2,j)                    
            historydf = pd.DataFrame(historicos[nombres[i]].history, index=historicos[nombres[i]].epoch)
            plt.plot(ylim=(0, max(1, historydf.values.max())))
            plt.plot(historydf)
            loss = historicos[nombres[i]].history['loss'][-1]
            acc = historicos[nombres[i]].history['acc'][-1]
            plt.title(nombres[i] +'- Loss: '+ str(loss)+', Accuracy: '+str(acc))

        i += 1
    plt.show()
    plt.savefig('Compararcion-Historicos.png')

######Funciones de ejecución de distintos modos

def logistic_regressionPlay(X_train, X_test, Y_train, Y_test, graf = True):
    x_train = X_train.T
    y_train = Y_train.T
    x_test = X_test.T
    y_test = Y_test.T
    logistic_regression(x_train, y_train, x_test, y_test,1,100,graf) 


def keras1Play(X_train, X_test, Y_train, Y_test, graf = True):
    model = Sequential()
    model.add(Dense(64, input_dim=13, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    keras(X_train, X_test, Y_train, Y_test, model, graf, "Keras-1")


def keras2Play(X_train, X_test, Y_train, Y_test, graf = True):
    model = Sequential()
    model.add(Dense(4, input_dim=13, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    keras(X_train, X_test, Y_train, Y_test, model, graf, "Keras-2")
        
def keras3Play(X_train, X_test, Y_train, Y_test, graf = True):
    keras(X_train, X_test, Y_train, Y_test, 0,graf,"Keras-3")

def keras4Play(X_train, X_test, Y_train, Y_test, graf = True):
    model = Sequential()
    model.add(Dense(50, input_dim=13, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
    keras(X_train, X_test, Y_train, Y_test, model, graf,"Keras-4")


def playAll(X_train, X_test, Y_train, Y_test, graf = False):
    logistic_regressionPlay(X_train, X_test, Y_train, Y_test, graf)
    sklearnLogisticRegression(X_train, X_test, Y_train, Y_test, graf)
    keras1Play(X_train, X_test, Y_train, Y_test, graf)
    keras2Play(X_train, X_test, Y_train, Y_test, graf)
    keras3Play(X_train, X_test, Y_train, Y_test, graf)
    keras4Play(X_train, X_test, Y_train, Y_test, graf)

################################################


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

        elif op==4: #Regresion lineal manual
            logistic_regressionPlay(X_train, X_test, Y_train, Y_test, graf = True)

        elif op==5: #Sklearn
            print(colored("[Presición aprox: 79.12 %]", "red"))
            sklearnLogisticRegression(X_train, X_test, Y_train, Y_test, graf = True)
        
        elif op==6: #Keras Basic -Presición: 54.95 %
            print(colored("Keras con 4 capas densas:", "red"))
            print(colored("Nodos: [64-tanh][32-tanh][16-tanh][1-softmax]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            print(colored("[Presición aprox: 54.95 %]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                keras1Play(X_train, X_test, Y_train, Y_test, graf = True)
            else: print("No ejecutado")

        elif op==7: #Keras Presición: 78.02 %
            print(colored("Keras con 3 capas densas:", "red"))
            print(colored("Nodos: [4-tanh][2-tanh][1-sigmoid]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            print(colored("[Presición aprox: 78.02 %]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                keras2Play(X_train, X_test, Y_train, Y_test, graf = True)
            else: print("No ejecutado")
            
        elif op==8: # keras 82.41% - establecida por defecto
            print(colored("Keras con 3 capas densas:", "red"))
            print(colored("Nodos: [30-tanh][20-tanh][1-sigmoid]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            print(colored("[Presición aprox: 82.41 %]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                keras3Play(X_train, X_test, Y_train, Y_test, graf = True)
            else: print("No ejecutado")
                        
        elif op==9: #keras 85.71 %
            print(colored("Keras con 4 capas densas:", "red"))
            print(colored("Nodos: [50-sigmoid][10-sigmoid][200-tanh][1-sigmoid]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            print(colored("[Presición aprox: 85.71 %]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                keras4Play(X_train, X_test, Y_train, Y_test, graf = True)
            else: print("No ejecutado")
            
       
        elif op==10: #Comparar graficas de historicos y de Matris de confusión
            print(colored("Se van a ejecutar todos los modelos, y a continuación se mostrará la comparación de gráficas", "red"))
            print(colored("[Grafica de Matriz de Confusión]", "red"))
            print(colored("[Grafica de Presición Historica]", "red"))
            print(colored("[Grafica de Presición]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                playAll(X_train, X_test, Y_train, Y_test, graf = False)
                compararModelosConfusion()
                compararModelosHistoricos()
                compararAciertos()
            else: print("No ejecutado")

        elif op==0:
            print("Saliendo...")
            break

        else:
            print (colored("La opción elegida no es correcta...\n","red"))
        


############
main()  
############
