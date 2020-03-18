import numpy as np
import copy
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y)
    dot = np.dot(Aux.T, Aux)
    return dot / (2 * len(X))
    

def gradiente(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]

    sumat = [0.] * n
    for i in range(n):
        for j in range(m):
                sumat[i] += (np.dot(X[j],Theta) - Y[j]) * X[j, i]  # Calcula el sumatorio con esta ecuación
        NuevaTheta[i] -= (alpha / m) * sumat[i]
    return NuevaTheta
    
    
def descenso_gradiente(X, Y, theta, alpha):
    theta = gradiente(X,Y,theta,alpha)
    costes = coste(X,Y,theta)
    return theta,costes


def normalizar(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    normalX = (X - mu) / sigma 
    return (normalX, mu, sigma)


def calcularNormal(X,Y):
    X = np.hstack([np.ones((len(X), 1)), X])
    dot = np.linalg.inv(np.dot(X.T, X))
    return np.dot(np.dot(dot, X.T), Y)


def main():
    iteraciones = 1500
    alpha = [0.1, 0.3, 0.01, 0.03, 0.001, 0.003]

    datos = carga_csv('ex1data2.csv')
    X = datos[:, 0:-1]
    Y = datos[:, -1:]

    Xn, mu, sigma = normalizar(X)
    m = len(X)
    
    fig = plt.figure()
    ax = fig.gca()

    costes = []
    thetas = []
    i=0
    # añadimos una columna de 1's a la X
    Xn = np.hstack([np.ones([m, 1]), Xn])

    for a in alpha: 
        theta = np.array(np.ones((len(Xn[0])))).reshape(len(Xn[0]), 1)
        fig = plt.figure()
        ax = fig.gca()
        for i in range(iteraciones):
            theta, coste =  descenso_gradiente(Xn, Y, theta, a)
            ax.plot(i, coste, 'bx')
        thetas.append(theta)
        costes.append(coste)
        plt.savefig('coste-'+str(a)+'.png')
        print("Alpha: "+str(a)+ " Coste: "+str(coste)+" - theta: "+str(theta))
        fig.clf()

    thetaFinal= thetas[np.argmin(costes)]
    costeFinal= np.min(costes)
    aFinal= alpha[np.argmin(costes)]
    print("[FINAL] Alpha: "+str(aFinal)+ " Coste: "+str(costeFinal)+" - Theta: "+str(thetaFinal))

    xn = np.array([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]])  # Genera un neuvo dato normalizado
    yd = np.dot(xn.T, thetaFinal)
    ThetaFormula = calcularNormal(X, Y)
    x = np.array([1, 1650, 3])
    yf = np.dot(x.T, ThetaFormula)

    print("Una casa con una superficie de 1.650 pies cuadrados y 3 habitaciones costará: ")
    print("[DESCENSO] "+str(round(yd[0],2))+" euros.")
    print("[FORMULA]  "+str(round(yf[0],2))+" euros.")
   
################
main()


