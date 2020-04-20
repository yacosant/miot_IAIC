import helper as help
from sklearn.datasets import make_classification, make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense

GRAF_DEBUG = False

def problema0():
    X, y = make_classification(n_samples=1000, n_features=2,
                            n_redundant=0, n_informative=2, random_state=7, n_clusters_per_class=1)
    help.plot_data(X,y)
    lr = LogisticRegression()
    lr.fit(X,y)
    help.plt.show()

def problema1():
    ep=50

    X, y = make_classification(n_samples=1000, n_features=2,
                            n_redundant=0, n_informative=2, random_state=7, n_clusters_per_class=1)

    help.plot_data(X,y)
    help.plt.savefig("problema1-"+str(ep)+"-graf0.png")
    #help.plt.show()

    model = Sequential()
    model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(x=X, y=y, verbose=0, epochs=ep)
    help.plot_loss_accuracy(history)
    help.plt.close(2)
    help.plt.savefig("problema1-"+str(ep)+"-graf1.png")

    help.plot_decision_boundary(lambda x: model.predict(x), X, y)
    help.plt.savefig("problema1-"+str(ep)+"-graf2.png")
    help.plt.show()
    
def problema2():
    #cambiar epochs
    ep=50

    X, y = make_moons(n_samples=1000, n_features=2,
                            n_redundant=0, n_informative=2, random_state=7, n_clusters_per_class=1)

    help.plot_data(X,y)
    help.plt.savefig("problema1-"+str(ep)+"-graf0.png")
    #help.plt.show()

    model = Sequential()
    model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(x=X, y=y, verbose=0, epochs=ep)
    help.plot_loss_accuracy(history)
    help.plt.close(2)
    help.plt.savefig("problema1-"+str(ep)+"-graf1.png")

    help.plot_decision_boundary(lambda x: model.predict(x), X, y)
    help.plt.savefig("problema1-"+str(ep)+"-graf2.png")
    help.plt.show()


def main():
    #Poner variable a True para mostrar las gráficas iniciales
    GRAF_DEBUG=True
    #0 - Regresión logistica de dos dimensiones (sin keras)
    #problema0()

    #1 - Regresión logistica de dos dimensiones (con keras)
    #problema1()

    #2 - Regresión logistica no lineal
    problema2()



main()
#comentar que en helper -> cambiado  plt.cm.RdYlBu  y todas las referencias similares
    # por -> plt.cm.get_cmap("RdYlBu")
