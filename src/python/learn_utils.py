from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sympy import Add
import numpy as np


def linear_combination(coef, basis, ndigits=4, sort=False):
    '''Sum c_i * f_i'''
    assert len(coef) == len(basis)

    cf_pairs = zip(coef, basis)

    if sort:
        cf_pairs = sorted(cf_pairs, key=lambda p: abs(p[0]), reverse=True)

    args = (c*f if abs(round(c, ndigits)) > 0 else 0 for c, f in cf_pairs)
    for arg in args:
        print '\t', arg


def plot_learning_curve(model, X, y):
    '''Progress of model error when train on 1 to all data'''
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))

        y_val_predict = model.predict(X_val)
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    fig = plt.figure()
    ax = fig.gca()
    ax.semilogy(np.sqrt(train_errors), 'rx', linestyle='--', label='train') 
    ax.semilogy(np.sqrt(val_errors), 'bx', linestyle='--', label='val')
    plt.legend(loc='best')
    return ax

                            
