
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

def pintaFun(puntosX, puntosFx, combined): 
    # Plot the data
    print(puntosFx)
    plt.plot(puntosX,puntosFx,  label='Función') 
    plt.plot(combined, color='red',lw=0,marker='+',markersize=10)
    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

def integra_mc(fun, a, b, num_puntos=10000):
    
    """
    sizes = np.linspace(a,b, num_puntos) 
   
    for size in sizes: 
        puntosX = np.random.randint(a, b, int(size)) #np.random.uniform(1, 100, int(size)) 
    """ 
    
    #generar coordenadas X entre a y b
    #puntosX = np.random.randint(a,b,size=num_puntos)
    puntosX = np.arange(a, b, 1)
    print(puntosX)

    #crea las Y de f(x)
    puntosFx = fun(puntosX)

    print(puntosFx)
    #obtiene minimo y maximo de f(x)
    min = 0 #np.amin(puntosFx)
    max = np.amax(puntosFx)
    print("[INFO] Minimo: "+ str(min) + " / Maximo: "+str(max))
    
    """
    for size in sizes: 
        puntosY = np.random.randint(min, max, int(size)) #same
    """
    tope= int(num_puntos/(b-a))
    for i in range(0, tope):
        puntosY = np.random.randint(min,max,99) #????
    #generar Y entre min y max
    #puntosY = np.random.randint(min,max,99)

    #devulve que puntos cumplen y<f(x)
    mask = (puntosY < puntosFx)
    # para pintar los superiores distintos sup = (puntosY => puntosFx)
    nPuntosDebajo =  np.sum(mask)
    print("Numero de puntos por debajo: "+str(nPuntosDebajo)) 
    
    #combina coordenadas x e i
    combined = np.vstack((puntosY, puntosX)).T

    #pinta puntos f(x)
    pintaFun(puntosX, puntosFx, puntosY)

    return 0


def main ():
    print("Start!")
    np.set_printoptions(threshold=sys.maxsize)
    #integra_mc(funcion,0, 30)
    #matriz = np.random.randint(0,2,3,4(1,1))
    integra_mc(funcion, 1, 100, 10000)
    
    print("End!")


main()
