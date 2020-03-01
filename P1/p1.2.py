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

def pinta(puntosX, puntosY):
    plt.scatter(puntosX, puntosY, marker='+', color = "red")

def coste(X, Y, Theta):
    """
    print("teta:"+str(Theta))
    print("x:"+str(X))
    print("y:"+str(Y))
    """
    
    """
    m = len(X)
    coste = (1 / (2 * m)) * np.dot((np.dot(Theta, X) - Y).T, np.dot(Theta, X) - Y) 
    print(coste)
    return coste"""
    #"""mi codigo:
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))
    #"""
    

def gradiente(X, Y, Theta, alpha):
    
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H - Y)
    for i in range(n):#igual a for n in range(columns):
        Aux_i = Aux * X[:, i]
        NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
    return NuevaTheta
    
    
def descenso_gradiente(X, Y, theta, alpha):
    theta = gradiente(X,Y,theta,alpha)
    costes = coste(X,Y,theta)
    print("Coste: "+str(costes)+" - theta: "+str(theta))
    return theta,costes

def main():
    iteraciones = 1500
    alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

    datos = carga_csv('ex1data2.csv')
    X = datos[:, 0:-1]#[:, :-1]
    Y = datos[:, -1:]

    Xn, mu, sigma = normalizar(X)
    m = len(X)
    
    fig = plt.figure()
    ax = fig.gca() #???

    costes = []
    thetas = []
    i=0
    
    for a in alpha: 
        # añadimos una columna de 1's a la X
        Xn = np.hstack([np.ones([m, 1]), Xn])
        theta = np.array(np.ones((len(Xn[0])))).reshape(len(Xn[0]), 1)

        fig = plt.figure()
        ax = fig.gca()
        for i in range(iteraciones):
            #sumatorio = np.array([])
            print(i)
            theta, coste =  descenso_gradiente(Xn, Y, theta, a)
            ax.plot(i, coste, 'bx')
            #plt.savefig('coste-'+str(alpha)+'.png')
            #print(theta)
        plt.savefig('coste-'+str(a)+'.png')
        thetas.append(theta)
        costes.append(coste)
        fig.clf()

    """
    plt.plot(X,theta[0] + theta[1]*X,color='blue') #recta definida por las thetas
    plt.savefig('p1.2-puntosYrecta.png')
    plt.show()
    """

    """
    Theta0, Theta1, Coste = make_data([-10,10], [-1,4], Xn, Y,theta[0],theta[1])   
    #Contorno
    plt.figure()
    plt.contour(Theta0, Theta1, Coste, np.logspace(-2, 3, 20), colors='blue')  
    plt.plot(theta[0],theta[1], marker='+',color = "red")
    plt.savefig('p1.2-contorno.png')
    plt.show()

    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Theta0, Theta1, Coste, linewidth=0, antialiased=False,cmap=cm.coolwarm)
    # Customize the z axis.
    ax.set_zlim(0,700)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax.set_xlim(-10,10)
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

	# Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('p1.2-3d.png')
    plt.show()
    """


def make_data(t0_range, t1_range, X, Y,t1,t2):
    """Genera las matrices X,Y,Z para generar un plot en 3D
    """
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)
    # Theta0 y Theta1 tienen las misma dimensiones, de forma que
    # cogiendo un elemento de cada uno se generan las coordenadas x,y
    # de todos los puntos de la rejilla
    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])

    return (Theta0, Theta1, Coste)

def normalizar(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    normalX = (x - mu) / sigma 
    return (normalX, mu, sigma)

main()


