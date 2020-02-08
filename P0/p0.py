
"""
generar random entre a y b
pillar el punto maximo

ver que porcentaje de ese area esta debajo de la funcion.
disparar puntos aleatorios entre a y b y entre min y maximo

la proporcion de puntos que caigan por debajo de la función será la proporcion de debajo de la función
(y generda menor de f(x) )
"""
#https://es.stackoverflow.com/questions/139223/se-puede-pasar-una-funci%C3%B3n-como-par%C3%A1metro-en-python

import time 
import numpy as np
import sys
import matplotlib.pyplot as plt

def funcion (x):
    
    a = 1
    b = 0
    c = 0
    
    #x = np.arange(-10, 10, 0.1)
    return ((a * x) ** 2) + (b * x) + c
    #return x*2

def pintaFun(puntosX, puntosY, x, y, encima, debajo): 
    plt.scatter(x[debajo], y[debajo], marker='+', color = "green")
    plt.scatter(x[encima], y[encima], marker='+',color = "grey")
    plt.plot(puntosX, puntosY, color = "blue")
    plt.show()


def integra_mc(fun, a, b, num_puntos=10000):
    #num_puntos=1000 #para que tarde menos. BORRAR para entrega
    
    #generar coordenadas X entre a y b
    puntosX = np.arange(a, b, 0.01)

    #crea las Y de f(x)
    puntosY = fun(puntosX) 

    #obtiene minimo y maximo de f(x)
    min = 0 
    max = np.amax(puntosY)
    print("[INFO] Minimo: "+ str(min) + " | Maximo: "+str(max))
    
    x =  a + (b - a)*np.random.random_sample(num_puntos)
    y = np.random.random_sample(num_puntos)*max

    debajo = y < fun(x)#np.where(y < fun(x))
    encima = y >= fun(x)#np.where(y >= fun(x))
    
    #Cuento cuantos puntos quedan por debajo de la funcion
    #print(debajo)
    #print(encima)
    nDebajo =  np.sum(debajo)
    nEncima =  np.sum(encima)
    nTotal = nDebajo+nEncima

    print("[INFO](Numero de puntos)  | Debajo: "+str(nDebajo)+" | Encima: "+str(nEncima)+" | Total: "+str(nEncima+nDebajo)+" |")
    
    #print("[INFO] Numero de puntos por debajo: "+str(nDebajo)) 
    #print("[INFO] Numero de puntos total: "+str(nTotal)) 

    area = (nDebajo/nTotal)*(b-a)*max
    print("[SOLUCION] Area por debajo: "+str((nDebajo/nTotal)*100)+"%") 
    print("[SOLUCION] Area por debajo: "+str(area)+" unidades") 
    #Pinto la grafica
    pintaFun(puntosX, puntosY, x, y, encima, debajo)

    return 0


def main ():
    print("[INICIO]")
    np.set_printoptions(threshold=sys.maxsize)
    integra_mc(funcion, 100, 1000, 10000)
    print("[FINAL]")


main()
