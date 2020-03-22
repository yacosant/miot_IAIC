import numpy as np
import copy
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt


def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    return valores.astype(float)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def pinta_frontera_recta(X, Y, theta):

    pinta_puntos(X,Y)
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
    np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)),
    xx1.ravel(),
    xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    # el cuarto parámetro es el valor de z cuya frontera se quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("frontera.png")
    plt.show()

def cost(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    return cost

def gradient(theta, XX, Y):
    H = sigmoid( np.matmul(XX, theta) )
    grad = (1 / len(Y)) * np.matmul(XX.T, H - Y)
    return grad


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
        plt.scatter(X[pos, 1], X[pos,2], marker=mark, c=cc)

def evaluaPorcentaje(X,Y,Theta): 
    cont = 0
    m = len(X)
    prediccion =1 / (1 + np.exp(-np.dot(Theta, X.T)))
    for i in range(m):
        if (prediccion.T[i] >= 0.5 and Y[i] == 1) or (prediccion.T[i] < 0.5 and Y[i] == 0):
            cont += 1
    print("Hay un "+ str((cont/m)*100) + "% de aciertos")

def main():
    datos = carga_csv('ex2data1.csv')
    X = datos[:, :-1]
    np.shape(X)         
    Y = datos[:, -1]
    np.shape(Y)         
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])

    initialTheta = np.zeros(3) 

    result = opt.fmin_tnc(func=cost , x0=initialTheta , fprime=gradient, args =(X, Y))

    print("Result :"+str(result))

    pinta_frontera_recta(X,Y,result[0])
    evaluaPorcentaje(X,Y,initialTheta)
main()