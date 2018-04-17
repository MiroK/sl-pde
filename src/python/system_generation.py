# Components for generating the candidates to form the system for sparse
# or symbolic learning
import sympy as sp


def multi_index(d, n):
    '''Multi indices of size n in d vars'''
    assert d > 0
    assert n >= 0

    if n == 0:
        return [tuple(0 for _ in range(d))]
    if d == 1:
        return [(n, )]
    
    return [(i, ) + index for i in range(n+1) for index in multi_index(d-1, n-i)]


def poly_indices(d, n):
    '''Polynomials up to degree d in n variables'''
    # This is useful for making polynomials as well for defining all the
    # derivatives of particular order
    assert 0 <= n
    assert 0 < d

    return sum((multi_index(d, size) for size in range(n+1)), [])


def polynomial(vars, multi_index):
    '''Symbolic polynomial; Prod x^index(k)_k for every k in index'''
    assert len(vars) == len(multi_index)
    assert all(i >= 0 for i in multi_index)

    if sum(multi_index) == 0: return 1

    p = sp.S(1)
    for var, exponent in zip(reversed(vars), multi_index):
        if exponent == 0:
            continue
        elif exponent == 1:
            p = p * var
        else:
            p = p * var**exponent
    return p


def polynomials(vars, n):
    '''A (monomial?) basis for polynomials of max degree n in the variables'''
    return [polynomial(vars, index) for index in poly_indices(len(vars), n)]


def Dt(f, order, t_var=sp.Symbol('t')):
    '''Expression for taking temporal derivative of f of order or up to [order]'''
    if isinstance(order, int):
        return sp.Derivative(f, t_var, order)

    return [Dt(f, t_var, o) for o in range(order[0]+1)]


def Dx(f, dim, order, x_vars=sp.symbols('x y z')):
    '''
    Expression for taking spatial derivaties of f : R^dim -> 1 
    of order or up to [order]
    ''' 
    if isinstance(order, int):
        indices = [sum(map(list, zip(x_vars, index)), []) for index in multi_index(dim, order)]
        return [sp.Derivative(f, *index) for index in indices]

    return sum((Dx(f, dim, o, x_vars) for o in range(order[0]+1)), [])


# Systematic way of having combinations to build the system
# Several components for systems -? return arrays?
# Build the system
