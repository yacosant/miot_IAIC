import numpy as np
import copy
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def pinta_frontera_recta(X, Y, theta):
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se
    # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    #plt.savefig("frontera.pdf")
    plt.show()
    plt.close()

def pinta_puntos(X,Y):
    plt.figure()
    mark='o'
    cc='g'
    i=0
    for i in range(2):
        pos= np.where(Y== i)
        if i==1:
            mark='+'
            cc='k'
        plt.scatter(X[pos, 0], X[pos,1], marker=mark, c=cc)
    plt.show()

def pinta_puntoss(X,Y):
    plt.figure()
    pos= np.where(Y== 1)
    plt.scatter(X[pos, 0], X[pos,1], marker='+', c='k')
    plt.show()

def main():
    datos = carga_csv('ex2data1.csv')
    X = datos[:, :-1]
    np.shape(X)         # (97, 1)
    Y = datos[:, -1]
    np.shape(Y)         # (97,)
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    pinta_puntos(X,Y)

main()