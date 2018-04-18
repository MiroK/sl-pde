from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sympy import Add
import numpy as np


def split(expr, at=Add):
    '''Break the expression AST into atoms taking `at` as node'''
    if expr.is_Symbol: return (expr, )

    if isinstance(expr, at):
        return sum((split(arg, at) for arg in expr.args), ())

    return (expr, )


def is_basis(collection):
    '''All guys are unique'''
    return len(set(collection)) == len(collection)


def allign(target, collection):
    '''Allign the collection tuple by target (when possible)'''
    assert is_basis(target) and is_basis(collection), (target, collection)
    
    if not target: return collection

    if not collection: return ()

    first, rest = target[0], target[1:]
    if first in collection:
        index = collection.index(first)
        return (first, ) + allign(rest, collection[:index] + collection[index+1:])

    return allign(rest, collection)


def compare(coef, basis, truth, ndigits=4):
    '''The idea here is to print side by side the expression build by
    regression and the truth
    '''
    assert len(coef) == len(basis)
    
    # Allign first the reg expression by the coef magnitude
    cf_pairs = sorted(zip(coef, basis), key=lambda p: abs(p[0]), reverse=True)
    # Expand back
    coef, basis = zip(*map(list, cf_pairs))
    # Allign the other guy by basis
    truth = split(truth)
    truth = allign(basis, truth)
    # Add some halos to truth if it is short
    if len(truth) < len(basis): truth = truth + (0, )*(len(basis) - len(truth))

    for c, f, true in zip(coef, basis, truth):
        print round(c, ndigits), '*', f, 'vs.', true


def linear_combination(coef, basis, sort=True):
    '''Sum c_i * f_i'''
    assert len(coef) == len(basis)
    # NOTE: sympy does simplication/expansion so it might be a bit difficult
    # to dig out the terms
    cf_pairs = sorted(zip(coef, basis), key=lambda p: abs(p[0]), reverse=True)

    return ' + '.join(['%s * (%s)' % cf for cf in cf_pairs])


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

                            
