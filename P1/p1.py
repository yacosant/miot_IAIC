import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def pinta(puntosX, puntosY):
    plt.scatter(puntosX, puntosY, marker='+', color = "red")
    #plt.scatter(x[encima], y[encima], marker='+',color = "grey")
    #plt.plot(puntosX, puntosY, color = "blue")
    #plt.savefig(dir+'-bucles.png') 
    plt.show()
    plt.clf()

def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

def gradiente(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H - Y)
    for i in range(n):
        Aux_i = Aux * X[:, i]
        NuevaTheta -= (alpha / m) * Aux_i.sum()
    return NuevaTheta

def descenso_gradiente(X, Y, teta, alpha):

    #m=10000
    m = np.shape(X)[0]

   
    """
    temp0 =0
    temp1 =0
    val = alpha*1/m

    #realizar sumatorio
    for i in range(m):
        temp0 += teta.dot(X) - Y
        temp1 += temp0 * X
    #actualizar valores
    teta[0]= teta[0]-val*temp0
    teta[1]= teta[1]-val*temp1
    """
    costes = coste(X,Y,teta)
    teta = gradiente(X,Y,teta,alpha)
    print("Coste: "+str(costes)+" - Teta: "+str(teta))

    return[teta,costes]


def main():
    datos = carga_csv('ex1data1.csv')
    X = datos[:, :-1]
    np.shape(X)         # (97, 1)
    Y = datos[:, -1]
    np.shape(Y)         # (97,)
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    #pinta(X,Y)
    # añadimos una columna de 1's a la X
    #print(X)
    #print("nuevo")
    X = np.hstack([np.ones([m, 1]), X])
    #print(X)
    alpha = 0.01
    teta = np.zeros(2)
    tetas, costes =  descenso_gradiente(X, Y, teta, alpha)
    tetas, costes =  descenso_gradiente(X, Y, tetas, alpha)
    """
    for i in range(1500):
        print(i)
        teta, costes =  descenso_gradiente(X, Y, teta, alpha)
        plt.scatter(i, costes, marker='x', color = "blue")
    """
    plt.show()

main()