import numpy as np
import copy
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values
    # suponemos que siempre trabajaremos con float
    return valores.astype(float)

def pinta(puntosX, puntosY):
    plt.scatter(puntosX, puntosY, marker='+', color = "red")
    #plt.scatter(x[encima], y[encima], marker='+',color = "grey")
    #plt.plot(puntosX, puntosY, color = "blue")
    #plt.savefig(dir+'-bucles.png') 
    #plt.show()
    #plt.clf()

def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

def gradiente(X, Y, Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H - Y)
    for i in range(n):
        Aux_i = Aux * X[:, i]
        NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
    return NuevaTheta

def descenso_gradiente(X, Y, teta, alpha):
    print("[descenso_gradiente] Teta in: "+str(teta))
    #m=10000
    m = np.shape(X)[0]

   
    """
    temp0 =0
    temp1 =0
    val = alpha*1/m

    #realizar sumatorio
    for i in range(m):
        temp0 += teta.dot(X) - Y
        temp1 += temp0 * X
    #actualizar valores
    teta[0]= teta[0]-val*temp0
    teta[1]= teta[1]-val*temp1
    """
    teta = gradiente(X,Y,teta,alpha)
    costes = coste(X,Y,teta)
    print("Coste: "+str(costes)+" - Teta: "+str(teta))
    print("[descenso_gradiente] Teta OUt: "+str(teta))
    return teta,costes

#def pintarLinea() :
  #  plt.plot(X,teta[0] + teta[1]*X, linestyle='-',color='blue')

def main():
    iteraciones = 1500
    fin = False

    datos = carga_csv('ex1data1.csv')
    X = datos[:, :-1]
    np.shape(X)         # (97, 1)
    Y = datos[:, -1]
    np.shape(Y)         # (97,)
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    pinta(X,Y)
    # añadimos una columna de 1's a la X
    #print(X)
    #print("nuevo")
    X = np.hstack([np.ones([m, 1]), X])
    #print(X)
    alpha = 0.01
    tetas = np.zeros(2)
    i=0

    #for i in range(15):
    while i < iteraciones : #and not fin:
        print(i)
        tempTetas = copy.deepcopy(tetas)

        #print("[MAIN] Teta IN: "+str(tetas))
        tetas, costes =  descenso_gradiente(X, Y, tetas, alpha)
        fin = np.array_equal(tempTetas, tetas)
        print(tempTetas)
        print(tetas)
        print("fin="+str(fin))
        i+=1
        plt.plot(X,tetas[0] + tetas[1]*X,color='blue')
        #print("[MAIN] Teta OUT: "+str(tetas))
        #plt.scatter(i, costes, marker='x', color = "blue")
    

    H1 = np.dot(X[0], tetas)
    H2 = np.dot(X[np.shape(X)[0]-1], tetas)

    print(H1)
    print(X[0][1])
    print(".----")
    print(H2)
    #plt.scatter(np.array([X[0],H1]), np.array([X[np.shape(X)[0]-1],H2]))
    #plt.scatter(X[0],H1 ) #, [X[np.shape(X)[0]-1],H2])
    
    #profesor plt.plot([X[0][1],X[np.shape(X)[0]-1][1]], [H1,H2], label='linear')
    plt.plot(X,tetas[0] + tetas[1]*X,color='blue')

    #plt.plot([X[0],H1], [X[np.shape(X)[0]-1],H2], label='linear')

    plt.show()

main()

