# Data from simulations are some values for each point in the spatial
# domain and time slices. This space-time domain is thus a hypercube.
# The idea here is to definesome functional representation of the data
# which can be evaluated, combined (+, -) and differentaited.

from numpy.polynomial.chebyshev import Chebyshev
import numpy as np


class GridFunction(object):
    '''
    Representation of a function defined on a Cartesian domain. The domain 
    is time interval x {spatial intervals}.
    '''
    def __init__(self, grid, data, name='u'):
        '''Grid is a tensor product of intervals'''
        assert len(grid) == len(data.shape), (len(grid), len(data.shape))
        assert map(len, grid) == list(data.shape), (map(len, grid), list(data.shape)) 

        self.grid = grid
        self.dim = len(grid)  # The space time dimension
        self.data = data
        self.__name__ = name  # For expression printing

    def __call__(self, point):
        '''The value at a space-time point reffered to by its indices'''
        # NOTE: if meshgrid is used i,j indexing is needed
        assert isinstance(point, tuple)
        return self.data[point]

    
def diff(f, ntuple, interpolant=Chebyshev, width=5, degree=None):
    '''
    Given a grid function f we construct here a polynomial approximation 
    to partial derivative of f specified by ntuple. The approximation is 
    constructed using width points using and a polynomial of give degree.
    '''
    # Assert odd because then we get the stencil of widht centerer around
    # the point
    assert width > 0 and width % 2 == 1
    # Typically the interpolants are polynomial, width points allows
    # width-1 degree
    if degree is None: degree = width-1

    # Make sure that we are not over fitting
    assert 0 <= degree < width
    # And that the tuple is sensible
    assert len(ntuple) == f.dim
    assert all(t >= 0 for t in ntuple)

    # We are only after nonzero derivatives
    diffs = [(variable, power) for variable, power in enumerate(ntuple) if power > 0]
    # And if there are non we delagate the job to func evaluation
    if not diffs: return f

    # Polynomial approx is computed using a symmetric stencil of width around index
    stencil = lambda index, w=width/2: range(index-w, index+w+1)

    # Still, the function needs to be evaluated at dim^d space points
    # So here all but i is fixed at their p value and i runs over the stencil
    points = lambda p, i: [tuple(p[j] if j != i else stencil_p for j in range(len(p)))
                            for stencil_p in stencil(p[i])]
    # Now the functions we have are evaluated at index points and we
    # pretty much proceeed by doing things on lines/axis
    def diff_func(func, i, n, X=f.grid, d=degree):
        # Compute the n-derivative in i-th direction using degree
        # FIXME: approach that reuse interpolant = derivs up to ?
        def foo(p):
            assert isinstance(p, tuple)
            # Eval foo at the line where all by i-th coords are fixed
            # (at their p valeu) and i goes over the stencil
            y = np.array([func(point) for point in points(p, i)])
            # Get grid physical point in i direction for interpolation
            Xi = X[i]   
            index_i = p[i]
            x = Xi[stencil(index_i)]  

            # Now we can compute the interpolant of degree and take its
            # derivative at 'p' - it's only the Xi that matters
            return interpolant.fit(x, y, d).deriv(n)(Xi[index_i])
        return foo

    # I want the highest derivative to be computed from the data
    diffs = sorted(diffs, key=lambda p: p[1], reverse=True)

    # Annotate
    name = ''.join(['x_{%d}^{%d}' % d for d in diffs])
    name = r'(d %s / d %s)' % (f.__name__, name)
    
    func = f
    while diffs:
        index, power = diffs.pop()
        # At first we diff the function which use the grid data
        # For other directions we use the derivatives
        func = diff_func(func=func, i=index, n=power)
    
    func.__name__ = name
    return func


def add(f, g):
    op = lambda p, f=f, g=g: f(p) + g(p)
    op.name = '(%s + %s)' % (f.__name__, g.__name__)
    return op


def sub(f, g):
    op = lambda p, f=f, g=g: f(p) - g(p)
    op.__name__ = '(%s - %s)' % (f.__name__, g.__name__)
    return op

    
def prod(f, g):
    op = lambda p, f=f, g=g: f(p) * g(p)
    op.__name__ = '(%s * %s)' % (f.__name__, g.__name__)
    return op

    
def quotient(f, g):
    op = lambda p, f=f, g=g: f(p) / g(p)
    op.__name__ = '(%s / %s)' % (f.__name__, g.__name__)
    return op

    
def compose(f, g):
    op = lambda p, f=f, g=g: f(g(p))
    op.__name__ = '(%s o %s)' % (f.__name__, g.__name__)
    return op

    
def pow(f, index):
    op = lambda p, f=f, index=index: f(p)**index
    op.__name__ = '(%s)**%d' % (f.__name__, index)
    return op


def identity():
    op = lambda p: 1
    op.__name__ = '1'
    return op


def polynomial(vars, multi_index):
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


def poly_indices(d, n):
    '''Polynomials up to degree d in n variables'''
    assert 0 < n
    assert 0 < d

    if d == 1:
        return [(i, ) for i in range(n)]
    else:
        return [(i, ) + j for i in range(n) for j in poly_indices(d-1, n-i)]


def coordinate(grid, i):
    '''Grid function for p[i]'''
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

# TODO: docs, split grid_function.py algebra?
# Temporal derivatives
# Spatial derivatives
# Polynomials
# Systematic way of having combinations to build the system
# Several components for systems -? return arrays?

# Build the system

# --------------------------------------------------------------------

if __name__ == '__main__':

    t = np.linspace(0, 1, 10)
    x = np.linspace(2, 4, 40)

    grid = [t, x]

    foo_t = diff(coordinate(grid, 0), (0, 1))
    foo_x = coordinate(grid, 1)
    point = (3, 2)
    print (foo_t(point), foo_x(point)), t[point[0]], x[point[1]]


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
