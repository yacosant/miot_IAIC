import helper as help
from scipy.io import loadmat
from displayData import displayData
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


def problemaX(num, model, X, y, ver, ep, y_noCat, X_test, y_test):
    history = model.fit(x=X, y=y, verbose=0, epochs=ep)
    help.plot_loss_accuracy(history)
    help.plt.close(2)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf1.png")

    help.plot_confusion_matrix(model, X, y_noCat)
    help.plt.savefig("problema"+str(num)+"-"+str(ver)+"-graf3.png")
    
    pred= (model.predict(X)>0.5).astype(int)
    print(classification_report(y, pred))
    help.plt.show()

    [test_cost, test_acc] = model.evaluate(X_test, y_test,verbose=0)
    print("Evaluando la parte de Test: Coste = "+str(test_cost)+", Precision ="+str(test_acc*100))

def menu():
    print("---MENU---\n")
    print("Ejercicios:\n")
    print("(1) - [4-tanh]-[2-tanh]-[10-softmax]")
    print("(2) - [64-tanh]-[32-tanh]-[16-softmax]-[10-softmax]")
    print("(3) - [64-tanh]-[32-tanh]-[16-softmax]-[10-sigmoid]")
    print("(4) - [64-tanh]-[32-tanh]-[16-tanh]-[10-sigmoid]")
    print("(5) - [64-tanh]-[32-tanh]-[16-tanh]-[10-softmax]")
    print("(6) - [128-tanh]-[64-tanh]-[32-tanh]-[10-softmax]")
    print("(0) - Salir")

    return int(input("Seleccione ejercicio a calcular (numero):"))


def main():
    data = loadmat ('numbers.mat')
    y = data ['y']
    X = data ['X']
    y[y == 10] = 0

    # Show data
    sample = help.np.random.choice(X.shape[0], 100)
    fig, ax = displayData(X[sample, :])
    fig.savefig('numeros.png')

    X_train , X_test , y_train , y_test = train_test_split(X, y ,test_size=0.33 , stratify=y)
    
    y_cat = to_categorical(y)
    y_cat_train = to_categorical(y_train)
    y_cat_test = to_categorical(y_test)

    op = menu()
    if op==0:
        print("Saliendo...")
    elif op==1:
        # [4-tanh]-[2-tanh]-[10-softmax]
        model = Sequential()
        model.add(Dense(units=4, input_shape=(400,), activation='tanh'))
        model.add(Dense(units=2, activation='tanh'))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])

    elif op==2:
        # [64-tanh]-[32-tanh]-[16-softmax]-[10-softmax]
        model = Sequential()
        model.add(Dense(units=64, input_shape=(400,), activation='tanh'))
        model.add(Dense(units=32, activation='tanh'))
        model.add(Dense(units=16, activation='softmax'))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])

    elif  op==3:             
        # [64-tanh]-[32-tanh]-[16-softmax]-[10-sigmoid]
        model = Sequential()
        model.add(Dense(units=64, input_shape=(400,), activation='tanh'))
        model.add(Dense(units=32, activation='tanh'))
        model.add(Dense(units=16, activation='softmax'))
        model.add(Dense(units=10, activation='sigmoid'))
        model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
    
    elif  op==4:             
        # [64-tanh]-[32-tanh]-[16-tanh]-[10-sigmoid]
        model = Sequential()
        model.add(Dense(units=64, input_shape=(400,), activation='tanh'))
        model.add(Dense(units=32, activation='tanh'))
        model.add(Dense(units=16, activation='tanh'))
        model.add(Dense(units=10, activation='sigmoid'))
        model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])

    elif  op==5:             
        # [64-tanh]-[32-tanh]-[16-tanh]-[10-softmax]
        model = Sequential()
        model.add(Dense(units=64, input_shape=(400,), activation='tanh'))
        model.add(Dense(units=32, activation='tanh'))
        model.add(Dense(units=16, activation='tanh'))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])


    elif  op==6:             
        # [128-tanh]-[64-tanh]-[32-tanh]-[10-softmax]
        model = Sequential()
        model.add(Dense(units=128, input_shape=(400,), activation='tanh'))
        model.add(Dense(units=64, activation='tanh'))
        model.add(Dense(units=32, activation='tanh'))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
    
    problemaX(5,model, X_train, y_cat_train,op,100, y_train, X_test, y_cat_test)


main()
#importado display data


