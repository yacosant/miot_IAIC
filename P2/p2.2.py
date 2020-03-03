import numpy as np
import copy
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures


def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def pinta_frontera_recta(X, Y, theta):

    pinta_puntos(X,Y)
    #plt.figure()
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
    #plt.savefig("frontera.pdf")
    plt.show()
    #plt.close()

def cost(theta, X, Y, lam):
    # H = sigmoid(np.matmul(X, np.transpose(theta)))
    H = sigmoid(np.matmul(X, theta))
    cost = ((- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))) + ((lam / (2 * (len(X)))) * np.sum(theta[1:]**2))
    return cost

def gradient(theta, XX, Y, lam):
    H = sigmoid( np.matmul(XX, theta) )
    grad = (1 / len(Y)) * (np.matmul(XX.T, H - Y)) #= los que sean distintos de cero)
    grad[1:] +=  ((lam/len(Y)) * theta )
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
    #plt.show()

def evaluaPorcentaje(X,Y,Theta):    #cambiar funcion!!!
    """ mi intento de codigo
    m = len(X)
    X = np.hstack([np.ones((m, 1)), X])
    prediccion = 1 / (1 + np.exp(-np.dot(Theta, X.T)))

    unos = ((prediccion >= 0.5 and Y == 1) or (prediccion < 0.5 and Y == 0))
    unos.sum()
    print(str(unos.sum()*100/m)+"% de aciertos")
    """
    #codigo git:
    m = len(X)
    #X = np.hstack([np.ones((m, 1)), X])
    z=np.dot(Theta, X.T)
    pred =1 / (1 + np.exp(-z))
    count = 0

    for i in range(m):
        if (pred.T[i] >= 0.5 and Y[i] == 1) or (pred.T[i] < 0.5 and Y[i] == 0):
            count += 1

    print('Hay un {}% de aciertos'.format((count/m)*100))

def plot_decisionboundary(X, Y, theta, poly):
    plt.clf()
    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.savefig("boundary.pdf")
    plt.close()

def main():
    datos = carga_csv('ex2data2.csv')
    X = datos[:, :-1]
    np.shape(X)
    Y = datos[:, -1]
    np.shape(Y)
    #pinta_puntos(X,Y)
    m = np.shape(X)[0]
    # añadimos una columna de 1's a la X
    X = np.hstack([np.ones([m, 1]), X])
    #---
    ## utilizar para invocar a la función de optimización
    initialTheta = np.zeros(len(X[0]))#np.zeros(3) 
    lam = 1

    #coste = cost(initialTheta,X,Y, lam)
    #print("Coste"+ str(coste))

    #grad = gradient(initialTheta, X, Y)
    #print("Gradiente"+ str(grad))

    result = opt.fmin_tnc(func=cost , x0=initialTheta , fprime=gradient, args =(X, Y,lam))
    print("result:")
    print(result)
    Theta = result[0]
    pinta_puntos(X,Y)
    plt.show()

    poly = PolynomialFeatures(6)
    X = poly.fit_transform(X)
    plot_decisionboundary(X, Y, Theta, poly)
    poly.fit_transform(X)

    #pinta_frontera_recta(X,Y,result[0])
    #evaluaPorcentaje(X,Y,initialTheta)
main()

