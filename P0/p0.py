"""
    [IAAC] - PRACTICA 0
    YACO SANTIAGO PEREZ
"""

import time 
import numpy as np
import sys
import matplotlib.pyplot as plt

def funcion (x):
    a = 1
    b = 0
    c = 0
    return ((a * x) ** 2) + (b * x) + c
    #return x*2

def pintaFun(puntosX, puntosY, x, y, encima, debajo, mode, npuntos):
    dir='./result/'+str(npuntos) 
    plt.scatter(x[debajo], y[debajo], marker='+', color = "green")
    plt.scatter(x[encima], y[encima], marker='+',color = "grey")
    plt.plot(puntosX, puntosY, color = "blue")
    #plt.show()
    if mode==0:
        plt.savefig(dir+'-vectorizado.png') 
    else: 
        plt.savefig(dir+'-bucles.png') 
    plt.clf()

def integra_mc(fun, a, b, num_puntos=10000):
    return [integra_mc_mode(funcion, 100, 1000,0, num_puntos), integra_mc_mode(funcion, 100, 1000,1, num_puntos)]
    
def integra_mc_mode(fun, a, b, mode, num_puntos=10000):
    if mode==0:
        print("[MODE] -------------- VECTORIZADO ["+ str(num_puntos)+"] -------------- ")
    else: 
        print("[MODE] ----------------- BUCLES ["+ str(num_puntos)+"] ---------------- ")
    
    tic = time.process_time() 
    #generar coordenadas X entre a y b
    puntosX = np.arange(a, b, 0.01)

    #crea las Y de f(x)
    if mode==0:
        puntosY = fun(puntosX) 
    else: 
        puntosY=[]
        for i in puntosX:
            puntosY+=[fun(i)]
            
    #obtiene minimo y maximo de f(x)
    min = 0 
    max = np.amax(puntosY)
    print("[INFO] Minimo: "+ str(min) + " | Maximo: "+str(max))
    
    x =  a + (b - a)*np.random.random_sample(num_puntos)
    y = np.random.random_sample(num_puntos)*max

    if mode==0:
        debajo = y < fun(x)
        encima = y >= fun(x) 
    else: 
        debajo=[]
        encima=[]   
        for i in range(len(x)):
            debajo+=[y[i] < fun(x[i])]
            encima+=[y[i] >= fun(x[i])]
    
    #Cuento cuantos puntos quedan por debajo de la funcion
    nDebajo =  np.sum(debajo)
    nEncima =  np.sum(encima)
    nTotal = nDebajo+nEncima

    print("[INFO](Numero de puntos)  | Debajo: "+str(nDebajo)+" | Encima: "+str(nEncima)+" | Total: "+str(nEncima+nDebajo)+" |")

    area = (nDebajo/nTotal)*(b-a)*max
    print("[SOLUCION] Area por debajo: "+str((nDebajo/nTotal)*100)+"%") 
    print("[SOLUCION] Area por debajo: "+str(area)+" unidades") 

    toc = time.process_time() 
    tiempo= 1000 * (toc - tic) 
    print("[TIEMPO]: "+str(tiempo))

    #Pinto la grafica
    pintaFun(puntosX, puntosY, x, y, encima, debajo, mode, num_puntos)

    return tiempo

def compara_tiempos(num_puntos,a,b): 

    sizes = np.linspace(a, num_puntos, 20) 
    times_dot = [] 
    times_fast = [] 
    times = []
    for size in sizes: 
        times =integra_mc(funcion, a, b, int(size))
        times_fast += [times[0]]
        times_dot += [times[1]]
    
    plt.close() #cierra la figura previa, donde se generaban los puntos y la funcion

    plt.figure() 
    plt.scatter(sizes, times_dot, c='red', label='bucle') 
    plt.scatter(sizes, times_fast, c='blue', label='vector') 
    plt.legend() 
    plt.savefig('times.png') 
    plt.show()

def main ():
    #definicion de variables
    num_puntos=10000
    a=100
    b=1000
    ########################
    
    print("[INICIO]")
    np.set_printoptions(threshold=sys.maxsize)
    
    compara_tiempos(num_puntos,a,b)
    print("------------------------------------------------ ")
    print("[INFO]: Puedes consultar las graficas generadas con ambos metodos en 'num_puntos-vectorizado.png' y en 'num_puntos-bucles.png'")
    print("[INFO]: Puedes consultar la graficas generada de comparcion de tiempos en 'times.png'")
    print("[FINAL]")

main()
