import helper as help
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from sklearn.metrics import classification_report

def problemaX(num, model, X, y, ver, ep):
    help.plot_data(X,y)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf0.png")

    history = model.fit(x=X, y=y, verbose=0, epochs=ep)
    help.plot_loss_accuracy(history)
    help.plt.close(2)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf1.png")

    help.plot_decision_boundary(lambda x: model.predict(x), X, y)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf2.png")

    help.plot_confusion_matrix(model, X, y)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf3.png")
    
    pred= help.np.concatenate( (model.predict(X)>0.5), axis=0 ).astype(int)
    print(classification_report(y, pred, target_names=['0', '1']))
    help.plt.show()

def main():

    #1 - Regresión logistica de dos dimensiones (con keras)
    """
    X1, y1 = make_classification(n_samples=1000, n_features=2,
                            n_redundant=0, n_informative=2, random_state=7, n_clusters_per_class=1)
    model10 = Sequential()
    model10.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    model10.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    problemaX(1,model10, X1,y1,0,50)
    
    #2 - version 0 -Regresión logistica no lineal - lunas
    X2, y2 = make_moons(n_samples=1000, noise=0.05, random_state=0)
    
    model20 = Sequential()
    model20.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    
    model20.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    problemaX(2,model20, X2,y2,0,100)
    
    #2 - version 1 - Cambiando el modelo para adaptarlo a las lunas
    
    model21 = Sequential()
    model21.add(Dense(units=4, input_shape=(2,), activation='tanh'))
    model21.add(Dense(units=2, activation='tanh'))
    model21.add(Dense(units=1, activation='sigmoid'))

    #model21.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model21.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])
    problemaX(2,model21, X2,y2,1,100)
    
    #3 - version 0 -Regresión logistica no lineal - circulos
    X3, y3 = make_circles(n_samples=1000, noise=0.05, factor=0.3, random_state=0)
    
    model30 = Sequential()
    model30.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    
    model30.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    y_p30 = problemaX(3,model30, X3,y3,0,100)
    """
    #3 - version 1 -Regresión logistica no lineal - circulos
    X3, y3 = make_circles(n_samples=1000, noise=0.05, factor=0.3, random_state=0)
    
    model31 = Sequential()
    model31.add(Dense(units=4, input_shape=(2,), activation='tanh'))
    model31.add(Dense(units=2, activation='tanh'))
    model31.add(Dense(units=1, activation='sigmoid'))
    
    model31.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])
    problemaX(3,model31, X3,y3,0,100)



main()
#comentar que en helper -> cambiado  plt.cm.RdYlBu  y todas las referencias similares
    # por -> plt.cm.get_cmap("RdYlBu")
