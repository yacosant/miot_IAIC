import numpy as np
import copy
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import scipy.optimize as opt
from scipy.optimize import minimize
import math

from checkNNGradients import checkNNGradients
from displayData import displayData


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    ##### this section is identical to the cost function logic we already saw #####
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

    ##### end of cost function logic, below is the new part #####
    
    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)
        
        d3t = ht - yt  # (1, 10)
        
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)
        
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
    

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    #grad= gradiente(params, input_size, hidden_size, num_labels, X, y, learning_rate) #no funciona mi funcion gradiente
    J= cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)
    return J, grad
    
def cost(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, l):
    m = X.shape[0]
    theta1 = params_rn[0:(num_ocultas * (num_entradas + 1))].reshape(num_ocultas, (num_entradas + 1))
    theta2 = params_rn[(num_ocultas * (num_entradas + 1)):].reshape(num_etiquetas, (num_ocultas + 1))

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2) #Para calcular la h
    J=0
    """
    coste = -1 * (1 / m) * np.sum((np.log(h) * (y) + np.log(1 - h) *(1 - y))) +\
    (float(l) / (2 * m)) * (np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))
    """
        # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    #return coste+
        # add the cost regularization term
    J += (float(l) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

    return J


def gradiente(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, l):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
    delta1 = np.zeros((num_ocultas, num_entradas + 1))
    delta2 = np.zeros((num_etiquetas, num_ocultas + 1))
    m = len(X)

    for i in range(1, m):
        a3 = h(X[i], theta1, theta2)
        l3 = a3 - y[i]

        z2 = np.hstack([1, np.dot(theta1, X[i].T)])
        a2 = sigmoid(z2)
        l2 = (np.dot(theta2.T, l3) * dSigmoid(z2))
        l2 = l2[1:]

        l3 = l3.reshape(len(l3), 1)
        a2 = a2.reshape(len(a2), 1)
        theta_aux = theta2[:, :]
        theta_aux[:, 0] = 0
        delta2 += np.dot(l3, a2.T) + (l / m) * theta_aux		

        a1 = X[i]
        l2 = l2.reshape(len(l2), 1)
        a1 = a1.reshape(len(a1), 1)
        theta_aux = theta1[:, :]
        theta_aux[:, 0] = 0
        delta1 += np.dot(l2, a1.T) + (l / m) * theta_aux

    return np.concatenate((delta1.ravel(), delta2.ravel())) / m


def min_coste(num_entradas, num_ocultas, num_etiquetas, X, y, reg):
	initialTheta1 = pesosAleatorios(num_entradas, num_ocultas)
	initialTheta2 = pesosAleatorios(num_ocultas, num_etiquetas)
	params_rn = np.concatenate((initialTheta1.ravel(), initialTheta2.ravel()))

	result = opt.fmin_tnc(func=cost, x0=params_rn, fprime=gradiente, args=(num_entradas, num_ocultas, num_etiquetas, X, y, reg))
	theta1 = np.reshape(result[0][:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
	theta2 = np.reshape(result[0][num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
	return (theta1, theta2)


def pesosAleatorios(L_in, L_out):
	e = math.sqrt(6) / math.sqrt(L_in + L_out)
	pesos = 2 * e * np.random.rand(L_out, L_in + 1) - e
	return pesos


def evaluar(h, y):
	m = len(h.T)
	cont = 0
	for i in range(m):
		if (np.argmax(h.T[i]) + 1) == y[i, 0]:
			cont += 1
	print('Acierta el {}%\n'.format((cont/m)*100))

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def dSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def h(x, theta1, theta2):
	z2 = np.dot(theta1, x.T)
	a2 = sigmoid(z2)
	a2 = np.hstack([1, a2.T])
	z3 = np.dot(theta2, a2.T)
	a3 = sigmoid(z3)
	return a3

def getH(x, theta1, theta2):
	z2 = np.dot(theta1, x.T)
	a2 = sigmoid(z2)
	m = len(a2.T)
	a2 = np.hstack([np.ones((m, 1)), a2.T])
	z3 = np.dot(theta2, a2.T)
	return sigmoid(z3)

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
        
    a1 = X
    z2 = a1.dot(theta1.T) # 5000x401 * 401x25 = 5000x25
    #ones for the bias unit
    a2 = np.insert(sigmoid(z2), 0, values = np.ones(m), axis = 1)
    z3 = a2.dot(theta2.T) # 5000x26 * 26x10 = 5000x10
    h = sigmoid(z3) # = a3 = g(z3)

    return a1, z2, a2, z3, h

def costNN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    
   

    m = X.shape[0]
    J = 0
    X = np.hstack((np.ones((m, 1)), X))
    # y10 = np.zeros((m, num_etiquetas))
    """
    Theta1 = params_rn[:((num_entradas + 1) * num_ocultas)
             ].reshape(num_ocultas, num_entradas + 1)
    Theta2 = params_rn[((num_entradas + 1) * num_ocultas):].reshape(num_etiquetas, num_ocultas + 1)

    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m, 1)), a1))  # hidden layer
    a2 = sigmoid(a1 @ Theta2.T)  # output layer

    y10 = y

    coste = cost(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg)

    reg_J = coste + reg / \
            (2 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    # Implement the backpropagation algorithm to compute the gradients

    grad1 = np.zeros((Theta1.shape))
    grad2 = np.zeros((Theta2.shape))

    for i in range(m):
        xi = X[i, :]  # 1 X 401
        a1i = a1[i, :]  # 1 X 26
        a2i = a2[i, :]  # 1 X 10
        d2 = a2i - y10[i, :]
        d1 = Theta2.T @ d2.T * dSigmoid(np.hstack((1, xi @ Theta1.T)))
        grad1 = grad1 + d1[1:][:, np.newaxis] @ xi[:, np.newaxis].T
        grad2 = grad2 + d2.T[:, np.newaxis] @ a1i[:, np.newaxis].T

    grad1 = 1 / m * grad1
    grad2 = 1 / m * grad2

    grad1_reg = grad1 + \
                (reg / m) * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    grad2_reg = grad2 + \
                (reg / m) * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    #grad = np.hstack((grad1.ravel(order='F'), grad2.ravel(order='F'))) #sin regularizar
    grad = np.hstack((grad1_reg.ravel(order='F'), grad2_reg.ravel(order='F')))
    """

    reg_J, grad = backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg)
    return reg_J, grad

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

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3

    # Show data
    sample = np.random.choice(X.shape[0], 100)
    fig, ax = displayData(X[sample, :])
    fig.savefig('numeros.png')
    #plt.show()

    X = np.hstack([np.ones((len(X), 1)), X])  # Le añade una columna de unos a las x
    
    encoder = OneHotEncoder(sparse=False, categories='auto')
    y_cat = encoder.fit_transform(y)

    """
    theta_vec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    theta_vec = theta_vec.reshape((len(theta_vec), 1))
    params_rn = np.concatenate((theta1.ravel(), theta2.ravel()))
    print(theta1.shape, theta2.shape, theta_vec.shape)
    print(cost(theta_vec,num_entradas, num_ocultas, num_etiquetas, X, y_cat, l))
    print("---")
    res = minimize(fun=backprop, x0=params_rn, args=(num_entradas, num_ocultas, num_etiquetas, X, y_cat, l), 
                method='TNC', jac=True, options={'maxiter': 5000})
    print(res)
    print("---")
    J, grad = backprop(theta_vec, num_entradas, num_ocultas, num_etiquetas, X, y_cat, l)
    print(J)
    print( grad.shape)

    print("---")
    
    coste, grad = backprop(params_rn, len(X[0]) - 1, len(theta1), len(theta2), X, y_cat, 1)
    print(coste)
    print("\n----\n")
    print(grad)  
    """
    #checkNNGradients(cost, 0)
    
    init_epi1 = np.sqrt(6)/np.sqrt((num_entradas + 1) + (num_ocultas))
    init_epi2 = np.sqrt(6)/np.sqrt((num_ocultas + 1) + (num_etiquetas))

    theta1 = np.random.rand((num_ocultas), (num_entradas + 1)) * (2 * init_epi1) - init_epi1
    theta2 = np.random.rand((num_etiquetas), (num_ocultas + 1)) * (2 * init_epi2) - init_epi2
    theta_vec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    theta_vec = theta_vec.reshape((len(theta_vec), 1))

    theta_vec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    theta_vec = theta_vec.reshape((len(theta_vec), 1))
    #print(cost(theta_vec,num_entradas, num_ocultas, num_etiquetas, X, y_cat, l))
    
    """
    J, grad = backprop(theta_vec, num_entradas, num_ocultas, num_etiquetas, X, y_cat, l)
    print(J)
    print(grad.shape)
    """
    # minimize our cost function
    """
    res = minimize(fun=backprop, x0=theta_vec, args=( num_entradas, num_ocultas, num_etiquetas, X, y_cat, l), 
                    method='TNC', jac=True, options={'maxiter': 5000})
    print(res)
    """


    """
    theta1, theta2 = min_coste(len(X[0]) - 1, len(theta1), len(theta2), X, y_cat, 1)

    print(theta1)
    print(theta2)
    evaluar(getH(X, theta1, theta2), y)
    """
    
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

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3

    X = np.hstack([np.ones((len(X), 1)), X])  # Le añade una columna de unos a las x
    
    encoder = OneHotEncoder(sparse=False, categories='auto')
    y_cat = encoder.fit_transform(y)

    theta_vec = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    theta_vec = theta_vec.reshape((len(theta_vec), 1))
    params_rn = np.concatenate((theta1.ravel(), theta2.ravel()))
    #print(theta1.shape, theta2.shape, theta_vec.shape)
    print("COSTE:")
    print(cost(theta_vec,num_entradas, num_ocultas, num_etiquetas, X, y_cat, l))
    print("---")
    """
    res = minimize(fun=backprop, x0=params_rn, args=(num_entradas, num_ocultas, num_etiquetas, X, y_cat, l), 
                method='TNC', jac=True, options={'maxiter': 5000})
    print(res)
    print("---")
    """

#main()
a = checkNNGradients(costNN, 0)
print(a)
#mainTest()



