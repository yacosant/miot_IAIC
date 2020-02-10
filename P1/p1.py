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

def descenso_gradiente(X, Y, alpha):
    x = X


def main():
    datos = carga_csv('ex1data1.csv')
    X = datos[:, :-1]
    np.shape(X)         # (97, 1)
    Y = datos[:, -1]
    np.shape(Y)         # (97,)
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    pinta(X,Y)
    # añadimos una columna de 1's a la X
    X = np.hstack([np.ones([m, 1]), X])
    alpha = 0.01
    
    #Thetas, costes = descenso_gradiente(X, Y, alpha)

main()