import numpy as np
import copy
#from pandas.io.parsers import read_csv
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def cost(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    return cost

def gradient(theta, XX, Y):
    H = sigmoid( np.matmul(XX, theta) )
    grad = (1 / len(Y)) * np.matmul(XX.T, H - Y)
    return grad

def cost_regul(theta, X, Y, lam):
    m = len(X)
    H = sigmoid(np.dot(theta,X.T))
    part3= (np.sum(np.power(theta[1:], 2))*lam)/(2*m)
    part2= (np.log(1-H)).T*(1-Y)
    part1= (np.log(H)).T*Y

    return -1/m*(np.sum(part1 + part2)) + part3

def gradient_regul(theta, XX, Y, lam):
    H = sigmoid(np.dot(XX, theta))
    thetaNoZeroReg = np.insert(theta[1:], 0, 0)
    gradient =  (np.dot(XX.T, (H - Y)) + lam * thetaNoZeroReg) / len(Y) 
    return np.vstack(gradient)

def costeMinimo(XX, Y, lam):
    initialTheta = np.zeros(len(XX[0])) 
    result = opt.fmin_tnc(func=cost, x0=initialTheta, fprime=gradient, args=(XX, Y, lam))
    return result[0]

def oneVsAll(X, y, num_etiquetas, reg):
    thetas = np.zeros((num_etiquetas, len(X[0])))
    y[y == 10] = 0
    for e in range(num_etiquetas):
        if y == e:  yy = 1
        else: yy = 0
        thetas[e] = costeMinimo(X, yy, reg)  # Calcula el coste de que sea esa etiqueta
    return thetas

def clasificador(X, Thetas):
	res = [0] * len(Thetas)
	i = 0
	for t in Thetas:
		res[i] = sigmoid(np.dot(t, X.T))
		i += 1
	return np.argmax(res[:10])  

def exec(X, thetas,n):
    for i in range(n): 
        sample = np.random.choice(X.shape[0], 1)
        plt.imshow(X[sample, :].reshape(-1, 20).T)
        plt.axis('off')
        val = clasificador(X[sample, :], thetas)  
        plt.title('Se clasifica como: {}'.format(val))
        plt.savefig('ejemplo{}.png'.format(i)) 
        plt.show()

def evaluaPorcentaje(X,Y,Theta): 
    cont = 0
    m = len(X)
    prediccion =1 / (1 + np.exp(-np.dot(Theta, X.T)))
    for i in range(m):
        if (prediccion.T[i] >= 0.5 and Y[i] == 1) or (prediccion.T[i] < 0.5 and Y[i] == 0):
            cont += 1
    print("Hay un "+ str((cont/m)*100) + "% de aciertos")

def main():
    data = loadmat ('ex3data1.mat')
    y = data ['y']
    X = data ['X']

    sample=np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()


    thetas = oneVsAll(X, y, 10, 0.1)
    exec(X, thetas,10)
    evaluaPorcentaje(X,y,thetas)

    #repasar evalua evaluaPorcentaje
    #y onVsAll falla por comparar array con nuemeor. invesgar
#######
main()