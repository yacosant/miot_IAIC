import numpy as np
import copy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt
from keras.utils.np_utils import to_categorical
import math

from checkNNGradients import checkNNGradients
from displayData import displayData

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    #backprop devuelve el coste y el gradiente de una red neuronal de dos capas
    coste = cost(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg)
    grad = gradiente(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg)
    return (coste, grad)


def cost(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, l):
    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    m = len(X)
    cost1 = 0
    cost2 = 0
    cost3 = 0
    for i in range(m):
        for k in range(num_etiquetas):
            cost1 += y[i][k] * np.log(h(X[i], theta1, theta2)[k]) + (1 - y[i][k]) * np.log(1 - h(X[i], theta1, theta2)[k])
    cost1 = - cost1 / m

    if l != 0:
        for j in range(1, num_ocultas):
            for k in range(1, num_entradas):
                cost2 += (theta1[j][k])**2
        cost2 = (l * cost2) / (2 * m)
        for j in range(1, num_etiquetas):
            for k in range(1, num_ocultas):
                cost3 += (theta2[j][k])**2
        cost3 = (l * cost3) / (2 * m)

    coste= cost1 + cost2 + cost3
    print(coste)
    return cost1 + cost2 + cost3

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

def main():
    weights = loadmat('ex4weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']#Theta1 dimensión 25x401 ; #Theta2 dimensión 10x26
    data = loadmat ('ex4data1.mat')
    y = data ['y']
    X = data ['X']

    # Show data
    sample = np.random.choice(X.shape[0], 100)
    fig, ax = displayData(X[sample, :])
    fig.savefig('numeros.png')
    #plt.show()

    X = np.hstack([np.ones((len(X), 1)), X])  # Le añade una columna de unos a las x
    
    y_cat = to_categorical(y)  # Categoriza los datos
    y_cat = y_cat[:, 1:]  # Se busca que el 1 esté en la primera posición y el 0 en la última

    """
    params_rn = np.concatenate((theta1.ravel(), theta2.ravel()))
    coste, grad = backprop(params_rn, len(X[0]) - 1, len(theta1), len(theta2), X, y_cat, 1)
    print(coste)
    print("\n----\n")
    print(grad)
    checkNNGradients(backprop, 0)
    """

    theta1, theta2 = min_coste(len(X[0]) - 1, len(theta1), len(theta2), X, y_cat, 1)

    print(theta1)
    print(theta2)
    evaluar(getH(X, theta1, theta2), y)

main()

""""
ejecucion

0.7299757738180074
[[ 0.         -0.06022963  0.07767786 ... -0.0488952  -0.06724469
  -0.05508428]
 [ 0.         -0.05020155 -0.03376321 ...  0.09095387 -0.07272082
   0.07151297]
 [ 0.          0.06738311 -0.04707989 ...  0.10227578  0.05558539
  -0.07610806]
 ...
 [ 0.         -0.04695212 -0.02659424 ... -0.05207056  0.09242731
  -0.09156621]
 [ 0.         -0.06734897 -0.06548361 ... -0.09439851 -0.02560382
   0.08182273]
 [ 0.          0.09319373  0.10155563 ... -0.01211355 -0.09626511
   0.08226435]]
[[ 0.         -0.21356093  2.06382127  0.10595035 -1.08325461 -0.95687434
   1.97426337 -0.63798729 -0.27110651 -0.48910606 -0.84819977  0.77864476
  -1.25236648 -0.66001989 -2.1882055  -2.30498178  0.98248814 -3.25031594
  -0.58338118  2.33253369  0.38281905  0.88669611 -3.23974084 -0.1361234
   0.27595864  2.38362418]
 [ 0.          0.93018587  0.35627494 -1.21295969 -0.63514754  0.68638627
  -1.40154727  0.18440164 -3.85518137 -1.74585693  0.37117319 -3.03951172
   1.58945225  0.10085009  2.23684653 -2.55684218  0.47634321 -0.16760989
  -0.40743458  0.25929972 -0.72724385  0.62279186 -0.77945511  0.38168649
  -3.73527394  1.47227491]
 [ 0.          0.50946456 -1.06042747 -0.04800677 -1.86338776  0.33815056
   0.93463503 -0.36270656 -1.66568764 -3.0872052  -2.60309975  0.30310841
   0.85908058  0.45832115  1.80817831 -0.1760567   0.48125024 -2.13331218
   0.49208583  0.32669979 -2.99787847  0.39538375 -1.1329269   0.57046592
   2.24664899 -2.3693601 ]
 [ 0.         -0.6895878   0.63873336 -3.11683699  0.85599173  2.64829047
   2.23161731  2.96134987 -0.08173533  0.66707816  0.01104919 -0.42942198
  -1.16214778  0.81978982 -1.43714445 -1.18618988 -2.82644176  0.31450125
   0.47356408 -1.62510587 -0.71318207 -0.40908475 -0.32079174 -3.49713262
  -0.75298856 -1.27410867]
 [ 0.          1.1730753   1.49372681  2.09680696 -0.9641455   0.88714029
  -1.83108374 -1.17935628  2.79681755  1.07303821 -3.00741423 -0.15899838
  -0.51125181 -2.47634978 -1.80625822 -2.19248387  0.29995595  1.40356617
   0.21951871 -0.48277744 -1.02218069 -4.40312872  0.06537852  0.16058681
  -0.2438442  -0.05954678]
 [ 0.         -0.16759983 -0.41539299 -1.15936978 -1.32012459 -3.33244607
   0.97964922  0.86911175  0.35973219 -0.86415276  0.56489545 -2.92181603
   2.5609512  -1.23712915 -1.6386502   1.1419822  -0.89019975  1.41820817
   0.15162624  2.037318    0.61326719  0.028786   -2.42670578 -3.70126827
  -0.43865788  0.73487953]
 [ 0.         -2.21529157  2.44570533  1.61058952  1.0465987  -3.02725459
  -0.87424699  0.82626777 -0.17427882 -0.80360166  0.65300112  1.50360288
  -3.2614981  -0.10748151  1.18046399 -2.46603496 -1.28196231 -1.23573154
  -2.43779648 -2.09494385  0.44453727  0.08587548 -0.17884447  0.34358969
   1.04595477 -2.36591851]
 [ 0.         -4.24520946 -1.59992169  0.30983043 -2.20968466  0.14483119
  -0.63821641 -1.46506464 -3.04636915 -0.30352708  0.4706446   0.80037349
   2.19852363 -0.76293488 -3.69559323 -1.16855759 -0.18686093  1.65724961
   0.17635904 -1.91280528 -2.44352892  0.40192785 -0.38849373  2.13397993
   0.12786126  1.88404587]
 [ 0.          0.25378082 -3.2116257   2.44891774 -0.06321522  0.35909306
  -2.01307153 -0.13533853 -1.89412656 -0.82261155 -0.39772043  0.55779155
  -4.47328458 -0.79091672 -0.92549112  2.87888539 -2.14299409  0.00947578
  -1.78364993 -1.89724547 -0.26812422  0.89763081 -0.65372224 -0.63858799
   0.20119619  0.74725845]
 [ 0.         -1.77174906 -2.14643658 -2.083687   -1.72152424 -0.74099467
  -1.6389029  -1.66377888  2.49596985  0.27210406 -0.21451691 -1.81017833
   0.81566371  0.53608738  1.25259481  0.82116161  1.58035319 -1.65605408
  -2.55947393 -1.41234466  0.54427266 -3.52606384 -0.01997985  1.98630699
  -2.23577529 -1.94221429]]
Acierta el 93.62%

"""
