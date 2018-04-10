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
    def __init__(self, grid, data):
        '''Grid is a tensor product of intervals'''
        if isinstance(data, np.ndarray): data = [data]
        
        # Shape consistency of all the components
        for d in data:
            assert len(grid) == len(d.shape), (len(grid), len(d.shape))
            assert map(len, grid) == list(d.shape), (map(len, grid), list(d.shape)) 

        self.grid = grid
        self.dim = len(grid)  # The space time dimension
        self.data = data

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
        # Still a grid function like
        foo.dim = func.dim
        foo.grid = X
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
