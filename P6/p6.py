import helper as help
from sklearn.datasets import make_classification, make_moons, make_circles, make_classification
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense


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
    #numero de version 
    ver=2

    X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)

    help.plot_data(X,y)
    help.plt.savefig("problema2-"+str(ver)+"-graf0.png")
    #help.plt.show()

    model = Sequential()
    model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(x=X, y=y, verbose=0, epochs=100)
    help.plot_loss_accuracy(history)
    help.plt.close(2)
    help.plt.savefig("problema2-"+str(ver)+"-graf1.png")

    help.plot_decision_boundary(lambda x: model.predict(x), X, y)
    help.plt.savefig("problema2-"+str(ver)+"-graf2.png")

    help.plot_confusion_matrix(model, X, y)
    help.plt.savefig("problema2-"+str(ver)+"-graf3.png")
    help.plt.show()


def problemaX(num, model, X, y, ver):
    help.plot_data(X,y)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf0.png")

    history = model.fit(x=X, y=y, verbose=0, epochs=100)
    help.plot_loss_accuracy(history)
    help.plt.close(2)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf1.png")

    help.plot_decision_boundary(lambda x: model.predict(x), X, y)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf2.png")

    help.plot_confusion_matrix(model, X, y)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf3.png")
    help.plt.show()


def main():
    #generalizar modelos aqui y graficas y tal en el método

    #0 - Regresión logistica de dos dimensiones (sin keras)
    #problema0()

    #1 - Regresión logistica de dos dimensiones (con keras)
    X, y = make_classification(n_samples=1000, n_features=2,
                            n_redundant=0, n_informative=2, random_state=7, n_clusters_per_class=1)
    model10 = Sequential()
    model10.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    model10.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    problemaX(1,model10, X,y,0)


    #2 - Regresión logistica no lineal
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)
    model20 = Sequential()
    model20.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    model20.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    problemaX(2,model20, X,y,0)

    model21 = Sequential()
    model21.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
    model21.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    problemaX(2,model21, X,y,1)




main()
#comentar que en helper -> cambiado  plt.cm.RdYlBu  y todas las referencias similares
    # por -> plt.cm.get_cmap("RdYlBu")
