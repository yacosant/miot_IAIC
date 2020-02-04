
"""
generar random entre a y b
pillar el punto maximo

ver que porcentaje de ese area esta debajo de la funcion.
disparar puntos aleatorios entre a y b y entre min y maximo

la proporcion de puntos que caigan por debajo de la función será la proporcion de debajo de la función
(y generda menor de f(x) )
"""
#https://es.stackoverflow.com/questions/139223/se-puede-pasar-una-funci%C3%B3n-como-par%C3%A1metro-en-python

import time 
import numpy as np
import matplotlib.pyplot as plt

def funcion (x):
    return x*2

def integra_mc(fun, a, b, num_puntos=10000):
    #matriz = numpy.random.randint(min_val,max_val,(<num_rows>,<num_cols>))

    #recorrer la funcion entre a y b para sacar min y max
    #generar puntos x entre a y b
    #generar puntos y entre min y max

    #recorrer los num_puntos comprobando uno por uno si y<f(x)

    for i in range(a, b):
        print(i+"")
    return 0


def main ():
    print("Start!")
    #integra_mc(funcion,0, 30)
    #matriz = np.random.randint(0,2,3,4(1,1))
    print(matriz)
    print("End!")


main()
