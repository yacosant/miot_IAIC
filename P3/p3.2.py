import numpy as np
import copy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def exec(X, theta1, theta2): 
    m = len(X)
    a1 = np.hstack([np.ones((m, 1)), X])

    z2 = np.dot(theta1, a1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2.T])
    z3 = np.dot(theta2, a2.T)
    return sigmoid(z3) #a3

def evaluaPorcentaje(H,Y): 
    cont = 0
    m = len(Y)
    for i in range(m):
        if (np.argmax(H.T[i]) + 1) == Y[i, 0]:
            cont += 1
    print("Hay un "+ str((cont/m)*100) + "% de aciertos")

def main():
    weights = loadmat('ex3weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']# Theta1 es de dimensión 25x401 ; Theta2 es de dimensión 10x26

    data = loadmat ('ex3data1.mat')
    y = data ['y']
    X = data ['X']

    h = exec(X, theta1, theta2)
    evaluaPorcentaje(h, y)
#######
main()