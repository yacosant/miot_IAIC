import numpy as np
import copy
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
import math

from checkNNGradients import checkNNGradients
from displayData import displayData


def backprop(params, num_entradas, num_ocultas, num_etiquetas, X, y, l):
    print("Entra en backprop")
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    y = np.matrix(y)
    coste = 0
    grad= gradiente(params, num_entradas, num_ocultas, num_etiquetas, X, y, l) 
    coste= cost(params, num_entradas, num_ocultas, num_etiquetas, X, y, l)
    return coste, grad
    
def cost(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, l):
    m = X.shape[0]
    theta1 = params_rn[0:(num_ocultas * (num_entradas + 1))].reshape(num_ocultas, (num_entradas + 1))
    theta2 = params_rn[(num_ocultas * (num_entradas + 1)):].reshape(num_etiquetas, (num_ocultas + 1))
    
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2) #Para calcular la h
    coste=0
    for i in range(m):
        coste1 = np.multiply(-y[i,:], np.log(h[i,:]))
        coste2 = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        coste += np.sum(coste1 - coste2)
    
    coste = coste / m
    #Termino de regularizacion
    coste += (float(l) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    return coste


def gradiente(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, l):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
    delta1 = np.zeros((num_ocultas, num_entradas + 1))
    delta2 = np.zeros((num_etiquetas, num_ocultas + 1))
    m = X.shape[0]
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    for t in range(m):
        a1t = a1[t,:]
        z2t = z2[t,:]
        a2t = a2[t,:]
        ht = h[t,:]
        yt = y[t,:]
        
        d3t = ht - yt
        
        z2t = np.insert(z2t, 0, values=np.ones(1))
        d2t = np.multiply((theta2.T * d3t.T).T, dSigmoid(z2t))
        
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # añade el termino de regularizacion
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * l) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * l) / m

    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    return grad

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = X
    z2 = a1.dot(theta1.T) 
    a2 = np.insert(sigmoid(z2), 0, values = np.ones(m), axis = 1)
    z3 = a2.dot(theta2.T) 
    h = sigmoid(z3) # = a3 = g(z3)

    return a1, z2, a2, z3, h
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def dSigmoid(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def pesosAleatorios(L_in, L_out):
	e = math.sqrt(6) / math.sqrt(L_in + L_out)
	pesos = 2 * e * np.random.rand(L_out, L_in + 1) - e
	return pesos

def min_coste(num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    initialTheta1 = pesosAleatorios(num_entradas, num_ocultas)
    initialTheta2 = pesosAleatorios(num_ocultas, num_etiquetas)
    params_rn = np.concatenate((initialTheta1.ravel(), initialTheta2.ravel()))
    params = (np.random.random(size=num_ocultas * (num_entradas + 1) + num_etiquetas * (num_ocultas + 1)) - 0.5) * 0.25

    result = minimize(fun=backprop, x0=params_rn, args=( num_entradas, num_ocultas, num_etiquetas, X, y,reg), method='TNC', jac=True, options={'maxiter':70})
    print(result)
    theta1 = np.reshape(result.x[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(result.x[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
    return (theta1, theta2)

def evaluar(h, y):
    correct = [1 if a == b else 0 for (a, b) in zip(h, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('Acierta el '+str(accuracy * 100)+'%')


def main():
    weights = loadmat('ex4weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']#Theta1 dimensión 25x401 ; #Theta2 dimensión 10x26
    data = loadmat ('ex4data1.mat')
    y = data ['y']
    X = data ['X']
    # valores iniciales
    num_entradas = 400
    num_ocultas = 25
    num_etiquetas = 10
    l = 1
    
    encoder = OneHotEncoder(sparse=False, categories='auto')
    y_cat = encoder.fit_transform(y)
    t1, t2= min_coste(num_entradas, num_ocultas, num_etiquetas, X, y_cat, l)

    X = np.hstack([np.ones((len(X), 1)), X])  
    a1, z2, a2, z3, h = forward_propagate(X, t1, t2)
    yPred = np.array(np.argmax(h, axis=1) + 1)
    evaluar(yPred, y)

    
def mainTest():
    weights = loadmat('ex4weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']#Theta1 dimensión 25x401 ; #Theta2 dimensión 10x26
    data = loadmat ('ex4data1.mat')
    y = data ['y']
    X = data ['X']
    # valores iniciales
    num_entradas = 400
    num_ocultas = 25
    num_etiquetas = 10
    l = 1

    X = np.hstack([np.ones((len(X), 1)), X])
    encoder = OneHotEncoder(sparse=False, categories='auto')
    y_cat = encoder.fit_transform(y)

    theta_vec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    theta_vec = theta_vec.reshape((len(theta_vec), 1))
    print("COSTE:")
    print(cost(theta_vec,num_entradas, num_ocultas, num_etiquetas, X, y_cat, l))
    print("---")


#mainTest()
#a = checkNNGradients(backprop, 0)
#print(a)
main()


