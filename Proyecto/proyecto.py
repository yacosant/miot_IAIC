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

options = ["[0] - Salir", "[1] - Generar Gráficas", "[2] - Mostrar información de la muestra", "[3] - Mostrar elementos de la muestra","[4] - Ejecución:[Manual]  Regresión Logistica" ,"[5] - Ejecución:[Sklearn] Regresión Logistica", "[6] - Ejecución:[Keras1]  [64-tanh][32-tanh][16-tanh][1-softmax]", "[7] - Ejecución:[Keras2]  [4-tanh][2-tanh][1-sigmoid]", "[8] - Ejecución:[Keras3]  [30-tanh][20-tanh][1-sigmoid]", "[9] - Ejecución:[Keras4]  [50-sigmoid][10-sigmoid][200-tanh][1-sigmoid] - ¡MEJOR!", "[10] - Ejecutar y comparar todas:[Graficas de Precisión y de Confusión]"]
nombres =["Sklearns-RegresionLogistica", "Keras-basic"]
ejecuciones ={}
confusiones ={}
historicos ={}
nombres = []

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

###################Regresion logistica manual

def initialize(dimension):
    
    weight = np.full((dimension,1),0.01)
    #bias = 0.0
    return weight#,bias

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

    """
def forwardBackward(weight,bias,x_train,y_train):
    # Forward
    
    y_head = sigmoid(np.dot(weight.T,x_train) + bias)
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    
    # Backward
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}
    
    return cost,gradients
    """


def update(weight,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    
    #for each iteration, update weight values
    for i in range(iteration):
        cost,derivative_weight = forwardBackward(weight,x_train,y_train)
        weight = weight - learningRate * derivative_weight
        
        costList.append(cost)
        index.append(i)

    parameters = {"weight": weight}
    
    print("iteration:",iteration)
    print("cost:",cost)

    """
    plt.plot(index,costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    """
    return parameters, derivative_weight
"""
def update(weight,bias,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    
    #for each iteration, update weight and bias values
    for i in range(iteration):
        cost,gradients = forwardBackward(weight,bias,x_train,y_train)
        weight = weight - learningRate * gradients["Derivative Weight"]
        bias = bias - learningRate * gradients["Derivative Bias"]
        
        costList.append(cost)
        index.append(i)

    parameters = {"weight": weight,"bias": bias}
    
    print("iteration:",iteration)
    print("cost:",cost)

    plt.plot(index,costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients
"""

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

"""
def predict(weight,bias,x_test):
    z = np.dot(weight.T,x_test) + bias
    y_head = sigmoid(z)

    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction
"""


def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
    dimension = x_train.shape[0]
    #weight,bias = initialize(dimension)
    weight= initialize(dimension)
    
    #parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)
    parameters, derivative_weight = update(weight,x_train,y_train,learningRate,iteration)

    #y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction = predict(parameters["weight"],x_test)

    acc =(100 - np.mean(np.abs(y_prediction - y_test))*100)
    ejecuciones['Logic-regresion-manual'] = acc
    print(colored("Precisión Test: " +str(acc)+" %", "blue"))

########################################################

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
    nombres.append(name)
    #model.summary()
    score = model.evaluate(X_test, Y_test, verbose=0)
    acc=score[1]*100
    ejecuciones[name] = acc
    #print(colored("Presición entrenamiento: " +str(history.history['acc'])+" %", "blue"))
    print(colored("Presición test: " +str(acc)+" %", "blue"))
    help.plot_loss_accuracy(history)
    plt.savefig('history-'+name+'.png')
    if graf == True:
        plt.show()
    confusiones[name] = help.plot_confusion_matrix(model, X_train, Y_train)
    plt.savefig('confusion_matrix-'+name+'.png')
    if graf == True:
        plt.show()
    
def compararModelosConfusion():
    plt.figure(figsize=(24,12))
    plt.suptitle("Comparación Matrices de Confusión",fontsize=24)
    plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

    tam=len(confusiones)
    for i in range(tam):
        plt.subplot(2,3,i+1)
        plt.title(nombres[i])
        sns.heatmap(confusiones[nombres[i]],annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
    plt.show()


def compararModelosHistoricos():
    plt.figure(figsize=(24,12))
    plt.suptitle("Comparación Historicos",fontsize=24)
    plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

    tam=len(historicos)
    for i in range(tam):
        plt.subplot(2,3,i+1)
        #plt.title(nombres[i])
        
        historydf = pd.DataFrame(historicos[nombres[i]].history, index=historicos[nombres[i]].epoch)
        plt.plot(ylim=(0, max(1, historydf.values.max())))
        plt.plot(historydf)
        loss = historicos[nombres[i]].history['loss'][-1]
        acc = historicos[nombres[i]].history['acc'][-1]
        plt.title(nombres[i] +'- Loss: '+ str(loss)+', Accuracy: '+str(acc))
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

        elif op==4: #Regresion lineal manual
            x_train = X_train.T
            y_train = Y_train.T
            x_test = X_test.T
            y_test = Y_test.T
            logistic_regression(x_train, y_train, x_test, y_test,1,100)

        elif op==5: #Sklearn
            print("")
            print(colored("[Presición aprox: 79.12 %]", "red"))
            sklearnLogisticRegression(X_train, X_test, Y_train, Y_test)
        
        elif op==6: #Keras Basic -Presición: 54.95 %
            print(colored("Keras con 4 capas densas:", "red"))
            print(colored("Nodos: [64-tanh][32-tanh][16-tanh][1-softmax]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            print(colored("[Presición aprox: 54.95 %]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                model = Sequential()
                model.add(Dense(64, input_dim=13, activation='tanh'))
                model.add(Dense(32, activation='tanh'))
                model.add(Dense(16, activation='tanh'))
                model.add(Dense(1, activation='softmax'))
                model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
                keras(X_train, X_test, Y_train, Y_test, model, True, "Keras-1")
            else: print("No ejecutado")

        elif op==7: #Keras Presición: 78.02 %
            print(colored("Keras con 3 capas densas:", "red"))
            print(colored("Nodos: [4-tanh][2-tanh][1-sigmoid]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            print(colored("[Presición aprox: 78.02 %]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                model = Sequential()
                model.add(Dense(4, input_dim=13, activation='tanh'))
                model.add(Dense(2, activation='tanh'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
                keras(X_train, X_test, Y_train, Y_test, model, True, "Keras-2")
            else: print("No ejecutado")
        
        elif op==8: # keras 82.41% - establecida por defecto
            print(colored("Keras con 3 capas densas:", "red"))
            print(colored("Nodos: [30-tanh][20-tanh][1-sigmoid]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            print(colored("[Presición aprox: 82.41 %]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                keras(X_train, X_test, Y_train, Y_test, 0,True,"Keras-3")
            else: print("No ejecutado")
            
        elif op==9: #keras 85.71 %
            print(colored("Keras con 4 capas densas:", "red"))
            print(colored("Nodos: [50-sigmoid][10-sigmoid][200-tanh][1-sigmoid]", "red"))
            print(colored("Optimizador [adam] | Loss [binary_crossentropy]", "red"))
            print(colored("[Presición aprox: 85.71 %]", "red"))
            ok = input("Pulsa enter para ejecutar (o escribe otra cosa para no ejecutarlo) >>> ")
            if ok == '':
                model = Sequential()
                model.add(Dense(50, input_dim=13, activation='sigmoid'))
                model.add(Dense(10, activation='sigmoid'))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

                keras(X_train, X_test, Y_train, Y_test, model, True,"Keras-3")
            else: print("No ejecutado")

        elif op==11: #Ejecutar todos 
            print("")
            model = Sequential()
            model.add(Dense(50, input_dim=13, activation='sigmoid'))
            #model.add(Dense(100, activation='tanh'))
            model.add(Dense(10, activation='sigmoid'))
            model.add(Dense(200, activation='tanh'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

            keras(X_train, X_test, Y_train, Y_test, model, True,"k8")

        elif op==10: #Comparar graficas de historicos y de Matris de confusión
            print("")
            compararModelosConfusion()
            compararModelosHistoricos()

        elif op==0:
            print("Saliendo...")
            break
           


        else:
            print (colored("La opción elegida no es correcta...\n","red"))
        
        #Debug
        (historicos)
        print(nombres)


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