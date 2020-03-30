import numpy as np
import copy
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
	coste1 = -(1 / m) * np.sum(Y.T * np.log(sigmoid(np.matmul(X, theta))) + (1 - Y.T) * np.log(1 - sigmoid(np.matmul(X, theta))))
	coste2 = (lam / (2 * m)) * np.sum(theta[1:]**2)
	return coste1 + coste2

def gradient_regul(theta, XX, Y, lam):
    H = np.array([sigmoid(np.dot(XX, theta))]).T
    thetaNoZeroReg = np.insert(theta[1:], 0, 0)
    thetaNoZeroReg =np.array([np.hstack([0, theta[1:]])]).T
    gradient =  (np.dot(XX.T, (H - Y)) + lam * thetaNoZeroReg) / len(Y) 
    return gradient


def costeMinimo(XX, Y, lam):
    initialTheta = np.zeros(len(XX[0])) 
    result = opt.fmin_tnc(func=cost_regul, x0=initialTheta, fprime=gradient_regul, args=(XX, Y, lam)) #regularizado
    return result[0]

def oneVsAll(X, y, num_etiquetas, reg):
    thetas = np.zeros((num_etiquetas, len(X[0])))
    y[y == 10] = 0
    for e in range(num_etiquetas):
        yy = (y == e).astype('int')
        thetas[e] = costeMinimo(X, yy, reg)
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
        plt.title("Se clasifica como: "+str(val))
        plt.savefig("ejemplo"+str(i)+".png")
        plt.show()

def evaluaPorcentaje(X,Y,Theta): 
    cont = 0
    m = len(X)
    for i in range(m):
        clasificado = clasificador(X[i], Theta)
        if (clasificado == Y[i]):
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

#######
main()