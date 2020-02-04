
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
    return x*2


def integra_mc(fun, a, b, num_puntos=10000):
    #generar coordenadas X entre a y b
    puntosX = np.random.randint(a,b,num_puntos)
    print(puntosX)

    #crea las Y de f(x)
    puntosFx = fun(puntosX)

    #obtiene minimo y maximo de f(x)
    min=np.amin(puntosFx)
    max=np.amax(puntosFx)
    print("[INFO] Minimo: "+ str(min) + " / Maximo: "+str(max))

    #generar Y entre min y max
    puntosY = np.random.randint(min,max,num_puntos)

    #devulve que puntos cumplen y<f(x)
    

    return 0


def main ():
    print("Start!")
    np.set_printoptions(threshold=sys.maxsize)
    #integra_mc(funcion,0, 30)
    #matriz = np.random.randint(0,2,3,4(1,1))
    integra_mc(funcion, 1, 100, 10000)
    print("End!")


main()
