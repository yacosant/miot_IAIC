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
    m = len(X)
    H = sigmoid(np.dot(theta,X.T))
    part3= (np.sum(np.power(theta[1:], 2))*lam)/(2*m)
    part2= (np.log(1-H)).T*(1-Y)
    part1= (np.log(H)).T*Y

    return -1/m*(np.sum(part1 + part2)) + part3


def gradient(theta, XX, Y, lam):
    H = sigmoid(np.dot(XX, theta))
    thetaNoZeroReg = np.insert(theta[1:], 0, 0)
    gradient =  (np.dot(XX.T, (H - Y)) + lam * thetaNoZeroReg) / len(Y) 
    return np.vstack(gradient)


def costeMinimo(XX, Y, lam):
    initialTheta = np.zeros(len(XX[0])) 
    result = opt.fmin_tnc(func=cost, x0=initialTheta, fprime=gradient, args=(XX, Y, lam))
    return result[0]


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
    #plt.show()

def plot_decisionboundary(X, Y, theta, poly):
    #plt.clf()
    #plt.figure()
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
    plt.savefig("boundary.png")
    #plt.close()

def main():
    datos = carga_csv('ex2data2.csv')
    X = datos[:, :-1]
    np.shape(X)
    Y = datos[:, -1]
    np.shape(Y)
    pinta_puntos(X,Y)
    m = np.shape(X)[0]

    poly = PolynomialFeatures(6)
    X2 = poly.fit_transform(X)
    initialTheta = np.zeros(X2.shape[1]) #np.zeros(len(X[0]))#np.zeros(3) 
    lam = 1
    
    coste = cost(initialTheta,X2,Y, lam)
    print("Coste: "+ str(coste))
    gradiente  = gradient(initialTheta,X2,Y,lam)
    print("gradiente: "+ str(gradiente))

    teta = costeMinimo(X2,Y, lam)
    plot_decisionboundary(X2, Y, teta, poly)
    plt.show()
"""
    #grad = gradient(initialTheta, X, Y)
    #print("Gradiente"+ str(grad))

    result = opt.fmin_tnc(func=cost , x0=initialTheta , fprime=gradient, args =(X, Y,lam))
    print("result:")
    print(result)
    Theta = result[0]
    pinta_puntos(X,Y)
    #plt.show()

   
    plot_decisionboundary(X2, Y, result[0], poly)

    #pinta_frontera_recta(X,Y,result[0])
    #evaluaPorcentaje(X,Y,initialTheta)

    """
main()

