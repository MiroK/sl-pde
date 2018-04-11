# Data from simulations are some values for each point in the spatial
# domain and time slices. This space-time domain is thus a hypercube.
# The idea here is to definesome functional representation of the data
# over the domain which can be evaluated, combined (+, -) and differentaited.

from numpy.polynomial.chebyshev import Chebyshev
import numpy as np


class GenericGridFunction(object):
    '''
    Representation of a function defined on a Cartesian domain. The domain  
    is time interval x {spatial intervals}. A thing f is a grid function 
    if it

    i) f(p) maps a ntuple of indices to number(s)
    ii) has a grid
    '''
    def __init__(self, grid, f, value_shape):
        '''f : index(grid) -> R^value_shape'''
        self.grid = grid
        self.dim = len(grid)  # Dimension of space-time cylinder
        self.value_shape = value_shape
        self.f = f

    def __call__(self, point, y):
        '''The value at a space-time point reffered to by its indices'''
        # NOTE: if meshgrid is used i,j indexing is needed
        assert isinstance(point, tuple)
        y.ravel()[:] = self.f(point)   # Value from f


class GridFunction(GenericGridFunction):
    '''Grid function constructed from data over grid'''
    def __init__(self, grid, data):
        # A single npdarry is a scalar. For other shapes we pass a list
        # of list of ... ndarray
        if isinstance(data, np.ndarray): data = [data]

        # Now all the components mush mathch the grid shape
        def is_consistent(d, grid=grid):
            dim, sizes = len(grid), map(len, grid)
            if isinstance(d, np.ndarray):
                return dim == len(d.shape) and sizes == list(d.shape)

            return all(is_consistent(di, grid=grid) for di in d)
        assert is_consistent(data)
        # NOTE: maybe it would be easier to make data = np.array(data)
        
        # With okay shaped components we see if the data defines in a
        # sensible way a vector tensor ...
        def list_shape(l):
            if not isinstance(l, list) or not l: return ()

            items_shape = set(map(list_shape, l))
            shape = items_shape.pop()
            assert not items_shape

            return (len(l), ) + shape
        value_shape = list_shape(data)

        fshape = value_shape
        while len(fshape) > 1:
            data = sum(data, [])
            fshape = (fshape[0]*fshape[1], ) + fshape[2:]

        # Again, not sure if it helps memory but
        y = np.zeros(fshape)
        def f(p, data=data, y=y):
            # Fill by look up
            for i, d in enumerate(data):
                y[i] = d[p]
            return y

        GenericGridFunction.__init__(self, grid, f, value_shape)


def Coordinate(grid, i):
    '''Grid function for i'th coordinate'''
    assert 0 <= i < len(grid)
    shape = (1, )  # A scalar
    f = lambda p, grid=grid[i], i=i: grid[p[i]]
    return GenericGridFunction(grid, f, shape)


def SpaceTimeCoordinate(grid):
    '''Vector in space time index -> physical'''
    shape = (len(grid), )
    f = lambda p, grid=grid: grid[p]
    return GenericGridFunction(grid, f, shape)


def SpatialCoordinate(grid):
    '''Spatial axes'''
    shape = (len(grid)-1, )
    # Let's try to be more efficient
    y = np.zeros(shape)
    def f(p, grids=grid[1:], y=y):
        for i, pi in enumerate(p[1:]):
            y[i] = grids[i][pi]
        return y
    
    return GenericGridFunction(grid, f, shape)


def TemporalCoordinate(grid):
    '''Time axis'''
    return Coordinate(grid, 0)


def diff(f, ntuple, interpolant=Chebyshev, width=5, degree=None):
    '''
    diff(partial derivative) maps a grid function to a grid function of 
    the same shape. Derivative uses polynomial approximation with width 
    points using and a polynomial of give degree.
    '''
    assert isinstance(f, GenericGridFunction)
    
    # Assert odd because then we get the stencil of widht centered around
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
        # Prealoc output
        out = np.zeros(np.prod(f.value_shape))
        y = np.zeros((width, len(out)))

        # Compute the n-derivative in i-th direction using degree
        # FIXME: approach that reuse interpolant = derivs up to ?
        def foo(p, out=out, y=y):
            assert isinstance(p, tuple)
            # Eval foo at the line where all by i-th coords are fixed
            # (at their p valeu) and i goes over the stencil            
            for x, y_row in zip(points(p, i), y): func(x, y_row)

            # Get grid physical point in i direction for interpolation
            Xi = X[i]   
            index_i = p[i]
            x = Xi[stencil(index_i)]  

            # Now we can compute the interpolant of degree and take its
            # derivative at 'p' - it's only the Xi that matters
            p = Xi[index_i]
            out.ravel()[:] = [interpolant.fit(x, y[:, col], d).deriv(n)(p)
                              for col in range(y.shape[1])]
            
            return out
        # Still a grid function like
        return GenericGridFunction(func.grid, foo, func.value_shape)

    # I want the highest derivative to be computed from the data
    # NOTE this assumes that the underlying function is smooth enough to
    # allow for derivative order to be changed
    diffs = sorted(diffs, key=lambda p: p[1], reverse=True)
    
    func = f
    while diffs:
        index, power = diffs.pop()
        # At first we diff the function which use the grid data
        # For other directions we use the derivatives
        func = diff_func(func=func, i=index, n=power)
    return func
