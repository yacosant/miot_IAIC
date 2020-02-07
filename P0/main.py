import numpy as np
import sys
import matplotlib.pyplot as plt

def funcion (x):
    return x*5

def pintaFun(fX,puntos):

    plt.plot(fX, fX, label='linear')
    plt.plot(puntos, 'ro')

    plt.show()

def integra_mc(fun, a, b, num_puntos=10000):

    X = np.random.randint(a,b,size=num_puntos)


    FX = fun(X)

    min = np.amin(FX)
    max = np.amax(FX)

    Y = np.random.randint(min,max,num_puntos)

    mask = (Y < FX)
    nPuntosDebajo =  np.sum(mask)
    
    juntos = np.vstack((Y, X)).T

    pintaFun(FX, juntos)

    return 0


def main ():

    np.set_printoptions(threshold=sys.maxsize)
    #integra_mc(funcion,0, 30)
    #matriz = np.random.randint(0,2,3,4(1,1))
    integra_mc(funcion, 1, 100, 10000)
    


main()