# Here we define operations over grid functions, well any functions of
# one argument, which let us build expressions
from grid_function import diff as diff_grid


def coordinate(grid, i):
    '''Grid function for the i-th coordinate of p[i]'''
    assert 0 <= i < len(grid)
    # I would like to avoid having to create data so we fake it
    class FakeData(object):
        def __init__(self, grid, i):
            self.shape = tuple(map(len, grid))
            self.grid = grid[i]
            self.i = i
        def __getitem__(self, p):
            return self.grid[p[self.i]]
        
    return GridFunction(grid, FakeData(grid, i), name='x%d' % i)


def coordinates(grid):
    '''Functions for the coordinates'''
    return [coordinate(grid, i) for i in range(len(grid))]


def identity():
    '''Multiplicative identity'''
    op = lambda p: 1
    op.__name__ = '1'
    return op


def null():
    '''Additive identity'''
    op = lambda p: 0
    op.__name__ = '0'
    return op

# The other primitives in the system are grid functions, their derivatives
# and other lambdas. Now we build their combinations. Note that combinations
# cannot be diffed

def add(f, g):
    '''A plus'''
    op = lambda p, f=f, g=g: f(p) + g(p)
    op.name = '(%s + %s)' % (f.__name__, g.__name__)
    return op


def sub(f, g):
    '''A minus'''
    op = lambda p, f=f, g=g: f(p) - g(p)
    op.__name__ = '(%s - %s)' % (f.__name__, g.__name__)
    return op

    
def prod(f, g):
    '''A *'''
    op = lambda p, f=f, g=g: f(p) * g(p)
    op.__name__ = '(%s * %s)' % (f.__name__, g.__name__)
    return op

    
def quotient(f, g):
    '''A /'''
    op = lambda p, f=f, g=g: f(p) / g(p)
    op.__name__ = '(%s / %s)' % (f.__name__, g.__name__)
    return op

    
def compose(f, g):
    '''f o g'''
    op = lambda p, f=f, g=g: f(g(p))
    op.__name__ = '(%s o %s)' % (f.__name__, g.__name__)
    return op

    
def pow(f, index):
    '''f^index'''
    op = lambda p, f=f, index=index: f(p)**index
    op.__name__ = '(%s)**%d' % (f.__name__, index)
    return op

# Construction of some expressions

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
    '''Prod x^index(k)_k for every k in index'''
    assert len(vars) == len(multi_index)
    assert all(v >= 0 for v in vars)

    if sum(multi_index) == 0: return identity

    p = identity()
    for var, exponent in zip(vars, multi_index):
        if exponent == 0:
            continue
        elif exponent == 1:
            p = prod(p, var)
        else:
            p = prod(p, pow(var, exponent))
    return p


def polynomials(vars, n):
    '''A (monomial?) basis for polynomials of max degree n in the variables'''
    return [polynomial(vars, index) for index in poly_indices(len(vars), n)]


def Dt(f, order):
    '''Temporal derivative of f of order or up to [order]'''
    if isinstance(order, int):
        index = (order, ) + tuple(0 for _ in range(f.dim-1))
        return diff_grid(f, index)

    return [Dt(f, o) for o in range(order[0]+1)]


def Dx(f, order):
    '''Spatial derivaties of f of order or up to [order]'''
    if isinstance(order, int):
        indices = [(0, ) + index for index in multi_index(f.dim-1, order)]
        return [diff_grid(f, index) for index in indices]

    return sum((Dx(f, o) for o in range(order[0]+1)), [])
    

# TODO: docs, split grid_function.py algebra?
# Temporal derivatives
# Spatial derivatives
# Polynomials


# Systematic way of having combinations to build the system
# Several components for systems -? return arrays?
# Build the system

# --------------------------------------------------------------------

if __name__ == '__main__':
    from grid_function import GridFunction
    import numpy as np

    n = 4
    grid = [np.linspace(-1, 4, n),
            np.linspace(0, 2, n),
            np.linspace(0, 1, n),
            np.linspace(-2, 0, n)]
    T, X, Y, Z = np.meshgrid(*grid, indexing='ij')

    data = T*X**2 + X + Y + Z

    f = GridFunction(grid, data, 'f')

    print len(Dx(f, [2]))


    exit()
    import matplotlib.pyplot as plt
    from math import sin

    x = np.linspace(-1, 1, 100)
    f_values = np.sin(5*np.pi*x)
    
    x_index = np.arange(len(x))
    f = GridFunction([x], f_values, 'f')
    g = GridFunction([x], f_values, 'g')

    print polynomial([f, g], (1, 1)).__name__

    plt.figure()
    # Derivative
    plt.plot(x, 5*np.pi*np.cos(5*np.pi*x))

    dgrid_f = compose(sin, diff(f, (1, ), width=11))

    print dgrid_f.__name__
    # Only eval away from the boundary
    x_interior_index = x_index[6:-6]
    plt.plot(x[x_interior_index], map(lambda i: dgrid_f([i]), x_interior_index))

    plt.show()
