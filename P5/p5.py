import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
#import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

def pintar(X, y, theta = np.array(([0],[0])), reg = 0):
    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 1], y, s = 50,  linewidths = 1)
    plt.grid(True)
    plt.title('Datos del Agua')
    plt.xlabel('Cambio del nivel del agua (x)')
    plt.ylabel('Agua que ha desbordado la presa (y)')
    if theta.any() != 0:
        plt.plot(np.linspace(X.min(), X.max()), theta[0] + theta[1] * np.linspace(X.min(), X.max()),  color='red', label = 'Optimized linear fit')
        plt.title('Datos del Agua: Linear Fit')
        
    plt.legend()
    #plt.show()

def coste(X, y, theta):
    h = np.dot(X, theta) 
    tmp = (h-y)** 2
    return tmp.sum()/(2*len(X))


def gradiante(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)  
    return inner / m

def coste_regularizado(theta, X, y, l=1):
    m = X.shape[0]
    reg = (l / (2 * m)) * np.power(theta[1:], 2).sum()

    return coste(X, y, theta) + reg

def gradiente_regularizado(theta, X, y, l=1):
    m = X.shape[0]
    reg = theta.copy()  
    reg[0] = 0  

    reg = (l / m) * reg

    return gradiante(theta, X, y) + reg

def minTheta(theta, X, y, l = 0):
    return minimize(fun=coste_regularizado,x0=theta,args=(X, y, l),method='TNC',jac=gradiente_regularizado,options={'disp': True}).x

def pintarcurvaAprendizaje(theta, X, y, Xval, yval, reg = 0):
    m = y.size
    
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    
    example_num = np.arange(1, (X.shape[0] + 1))
    for i in np.arange(m):
        
        opt_theta = minTheta(theta, X[:i + 1], y[:i + 1], reg)
        error_train[i] = coste_regularizado(opt_theta, X[:i + 1], y[:i + 1], reg)
        error_val[i] = coste_regularizado(opt_theta, Xval, yval, reg)
    
    printarErroresCurvaAprendizaje(example_num, error_train, error_val, reg)
    return opt_theta

def printarErroresCurvaAprendizaje(example_num, error_train, error_val, reg):
    plt.figure(figsize = (12, 8))
    plt.plot(example_num, error_train, label = 'Error de Entrenamiento')
    plt.plot(example_num, error_val, label = 'Error de Validación cruzada')
    plt.title('Curva de aprendizaje: Sin Regularización')
    if reg != 0:
        plt.title('Curva de aprendizaje: Lambda = {0}'.format(reg))
    plt.xlabel('Número de ejemplo de entrenamiento')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def polyFeatures(X, p):
    for i in np.arange(p):
        dim = i + 2
        X = np.insert(X, X.shape[1], np.power(X[:,1], dim), axis = 1)
    
    X_norm = X
    #Normalizar
    means = np.mean(X_norm, axis=0)
    X_norm[:, 1:] = X_norm[:, 1:] - means[1:]
    stds = np.std(X_norm, axis = 0)
    X_norm[:, 1:] = X_norm[:, 1:] / stds[1:]
    
    return X, X_norm 

def plotFit(X, y, degree, num_points, reg = 0):
    X_poly = polyFeatures(X, degree)[1]
    starting_theta = np.ones((X_poly.shape[1], 1))
    opt_theta = minTheta(starting_theta, X_poly, y, reg)
    x_range = np.linspace(-55, 50, num_points)
    x_range_poly = np.ones((num_points, 1))
    x_range_poly = np.insert(x_range_poly, x_range_poly.shape[1], x_range.T, axis = 1)
    x_range_poly = polyFeatures(x_range_poly, len(starting_theta)-2)[0]
    y_range = x_range_poly @ opt_theta
    pintar(X, y)
    plt.plot(x_range, y_range,  color = "blue", label = "Polynomial regression fit")
    plt.title('Polynomial Regression Fit: No Regularization')
    if reg != 0:
        plt.title('Polynomial Regression Fit: Lambda = {0}'.format(reg))
    plt.legend()
    plt.show()

def main():
    dato = loadmat('ex5data1.mat')
    X, y, Xval, yval, Xtest, ytest = map(np.ravel, [dato['X'], dato['y'], dato['Xval'], dato['yval'], dato['Xtest'], dato['ytest']])
    X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
    
    #pintar(X,y)
    #plt.show()

    ###############
    print("################")
    lamda = 1
    theta = np.ones(X.shape[1]) #[1. 1.]
    cost=coste(X, y, theta)
    print("Coste: "+str(cost))
    g=gradiante(theta, X, y)
    print("Gradiente: "+str(g))
    print("################")

    ################ Regularizado
    cost=coste_regularizado(theta, X, y, lamda)
    print("Coste reg: "+str(cost))
    g=gradiente_regularizado(theta, X, y, lamda)
    print("Gradiente reg: "+str(g))
    print("################")

    ################ Linear Fit
    lamda=0
    theta = np.ones(np.shape(X)[1])
    theta_min = minTheta(theta,X, y, lamda)
    print("Theta: "+str(theta_min))
    pintar(X,y,theta_min,lamda)
    plt.show()
    print("################")

    ################ Curvas de aprendizaje
    pintarcurvaAprendizaje(theta, X, y, Xval, yval, reg = 0)

    ################ Regresión polinomial
    X_poly = polyFeatures(X, 8)[1]
    X_poly_val = polyFeatures(Xval, 8)[1]

    plotFit(X, y, 8, 1000, reg = 0)
    
    starting_theta = np.ones((X_poly.shape[1], 1))
    pintarcurvaAprendizaje(starting_theta, X_poly, y, X_poly_val, yval,0)
    
    #Lamda 1
    pintarcurvaAprendizaje(starting_theta, X_poly, y, X_poly_val, yval,1)
    #Lamda 100
    pintarcurvaAprendizaje(starting_theta, X_poly, y, X_poly_val, yval,100)
    
    ################ Descubrir valor optimo
    m = np.shape(X_poly)[0]
    ones = np.ones((m, 1))
    Xpoly = np.hstack((ones, X_poly))
    m = np.shape(X_poly_val)[0]
    ones = np.ones((m, 1))
    Xvalpoly = np.hstack((ones, X_poly_val))
    
    landaList = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    costeE, costeVal = [], []
    #starting_theta = np.ones((X_poly.shape[1], 1))
    #starting_theta = np.ones(np.shape(X_poly)[1])
    for l in landaList:
        res = minTheta(starting_theta,X_poly, y, l) 
        print("aqui res: "+str(res))
        tramic = coste( X_poly, y, res)
        costeE.append(tramic)
        validac = coste(X_poly_val, yval, res)
        costeVal.append(validac)    
    printarErroresCurvaAprendizaje(landaList, costeE, costeVal, 0)

    ############## Prueba para Xtest, ytest con lamda 3
    
    plotFit(Xtest, ytest, 8,1000,3)

    Xtest_poly = polyFeatures(Xtest, 8)[0]

    #Normalizar
    means = np.mean(X, axis=0)
    stds = np.std(X, axis = 0)

    Xtest_poly[:, 1:] = Xtest_poly[:, 1:] - means[1:]
    Xtest_poly[:, 1:] = Xtest_poly[:, 1:] / stds[1:]

    starting_theta = np.ones((Xtest_poly.shape[1], 1))
    res = minTheta(starting_theta,Xtest_poly, ytest, 3) 
    error_train = coste_regularizado(res, Xtest_poly, ytest, 3)
    print(error_train)
    pintarcurvaAprendizaje(res, Xtest_poly, ytest, X_poly_val, yval,3)
   

    ##############
main()