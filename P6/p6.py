import helper as help
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical

def problemaX(num, model, X, y, ver, ep, multi=False, yy=0):
    if multi==False:
        help.plot_data(X,y)
        help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf0.png")

    history = model.fit(x=X, y=y, verbose=0, epochs=ep)
    help.plot_loss_accuracy(history)
    if multi==False:
        help.plt.close(2)
    else:
        help.plt.close(1)
    
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf1.png")

    #IF segun si es multiclass
    if multi==False:
        help.plot_decision_boundary(lambda x: model.predict(x), X, y)
    else:
        help.plot_multiclass_decision_boundary(model, X, yy)
    
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf2.png")

    if num!= 1: 
        help.plot_confusion_matrix(model, X, yy)
        help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf3.png")

    if multi==False:
        pred= help.np.concatenate( (model.predict(X)>0.5), axis=0 ).astype(int)
    else:
        pred= (model.predict(X)>0.5).astype(int)

    print(classification_report(y, pred))
    help.plt.show()

def menu():
    print("---MENU---\n")
    print("Ejercicios:\n")
    print("(1) - [2 Dimensiones] Regresión logistica de dos dimensiones")
    print("(2) - [Lunas v1] ")
    print("(3) - [Lunas v2] - Clasifica bien")
    print("(4) - [Circulos v1] ")
    print("(5) - [Circulos v2] - Clasifica bien")
    print("(6) - [Espiral v1] ")
    print("(7) - [Espiral v2] - Clasifica bien")
    print("(0) - Salir")

    return int(input("Seleccione ejercicio a calcular (numero):"))


def main():
    op = menu()
    if op==0:
        print("Saliendo...")
    elif op==1:
        #1 - Regresión logistica de dos dimensiones (con keras)
        
        X1, y1 = make_classification(n_samples=1000, n_features=2,
                                n_redundant=0, n_informative=2, random_state=7, n_clusters_per_class=1)
        model10 = Sequential()
        model10.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
        model10.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        problemaX(1,model10, X1,y1,0,50)
    
    elif  op==2:
        #2 - version 0 -Regresión logistica no lineal - lunas
        X2, y2 = make_moons(n_samples=1000, noise=0.05, random_state=0)
        
        model20 = Sequential()
        model20.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
        
        model20.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        problemaX(2,model20, X2,y2,0,100)
    
    elif  op==3:
        #2 - version 1 - Cambiando el modelo para adaptarlo a las lunas
        
        model21 = Sequential()
        model21.add(Dense(units=4, input_shape=(2,), activation='tanh'))
        model21.add(Dense(units=2, activation='tanh'))
        model21.add(Dense(units=1, activation='sigmoid'))

        #model21.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        model21.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])
        problemaX(2,model21, X2,y2,1,100)

    elif  op==4:  
        #3 - version 0 -Regresión logistica no lineal - circulos
        X3, y3 = make_circles(n_samples=1000, noise=0.05, factor=0.3, random_state=0)
        
        model30 = Sequential()
        model30.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))

        model30.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        problemaX(3,model30, X3,y3,0,100)
    elif  op==5:        
        #3 - version 1 -Regresión logistica no lineal - circulos
        X3, y3 = make_circles(n_samples=1000, noise=0.05, factor=0.3, random_state=0)
        
        model31 = Sequential()
        model31.add(Dense(units=4, input_shape=(2,), activation='tanh'))
        model31.add(Dense(units=2, activation='tanh'))
        model31.add(Dense(units=1, activation='sigmoid'))
        
        model31.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])
        problemaX(3,model31, X3,y3,1,100)
        
    elif  op==6:       
        #4 - version 0 -Regresión logistica no lineal - Espiral
        X4, y4 = help.make_multiclass(K=3)
        
        model40 = Sequential()
        model40.add(Dense(units=4, input_shape=(2,), activation='tanh'))
        model40.add(Dense(units=2, activation='tanh'))
        model40.add(Dense(units=3, activation='softmax'))
        
        model40.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
        y4_cat = to_categorical(y4)
        problemaX(4,model40, X4,y4_cat,0,100, True,y4)

    elif  op==7:        
        #4 - version 1 -Regresión logistica no lineal - Espiral mejorada
        
        model41 = Sequential()
        model41.add(Dense(units=64, input_shape=(2,), activation='tanh'))
        model41.add(Dense(units=32, activation='tanh'))
        model41.add(Dense(units=16, activation='softmax'))
        model41.add(Dense(units=3, activation='softmax'))
        
        model41.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
        y4_cat = to_categorical(y4)
        problemaX(4,model41, X4,y4_cat,0,50, True,y4)

    else:
        print("Opcion erronea")


main()
#comentar que en helper -> cambiado  plt.cm.RdYlBu  y todas las referencias similares
    # por -> plt.cm.get_cmap("RdYlBu")
