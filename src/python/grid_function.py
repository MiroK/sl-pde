# Data from simulations are some values for each point in the spatial
# domain and time slices. This space-time domain is thus a hypercube.
# The idea here is to definesome functional representation of the data
# over the domain which can be evaluated, combined (+, -) and differentaited.

from numpy.polynomial.chebyshev import Chebyshev
from probe_io import (read_time, read_spatial_grid, read_probe_data,
                      write_structured_points)
import numpy as np
import os


class GenericGridFunction(object):
    '''
    Representation of a function defined on a Cartesian domain. The domain  
    is time interval x {spatial intervals}. A thing f is a grid function 
    if it

    i) f(p) maps a ntuple of indices to number(s)
    ii) has a grid
    '''
    def __init__(self, grid, f, value_shape=()):
        '''f : index(grid) -> R^value_shape'''
        self.grid = grid
        self.dim = len(grid)  # Dimension of space-time cylinder
        self.value_shape = value_shape
        self.f = f

    def __call__(self, point):
        '''The value at a space-time point reffered to by its indices'''
        # NOTE: if meshgrid is used i,j indexing is needed
        assert isinstance(point, tuple)
        return self.f(point)   # Value from f


class GridFunction(GenericGridFunction):
    '''Grid function constructed from data over grid'''
    def __init__(self, grid, data):
        # Reference to data for storing
        self.data = data

        # A single npdarry is a scalar.
        if isinstance(data, np.ndarray):
            GenericGridFunction.__init__(self, grid, f=lambda p: data[p], value_shape=())
            return None

        assert len(data) > 1  # Don't want len 1 vectors, or row vectors?
        # Everything else not is a tenseor valued

        # Now all the components mush mathch the grid shape
        def is_consistent(d, grid=grid):
            dim, sizes = len(grid), map(len, grid)
            print d, dim, sizes
            if isinstance(d, np.ndarray):
                return dim == len(d.shape) and sizes == list(d.shape)

            return all(is_consistent(di, grid=grid) for di in d)
        assert is_consistent(data)
        
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
            y.ravel()[:] = [d[p] for d in data]
            return y

        GenericGridFunction.__init__(self, grid, f, value_shape)
        
    def write(self, output, varname='', write_binary=True):
        '''Preliminary write the probe as VTK structured points'''
        base, ext = os.path.splitext(output)
        assert ext == '.vtk'
        
        time, grid = self.grid[0], self.grid[1:]

        header_data = {'origin': tuple(g[0] for g in grid),
                       'sizes': tuple(map(len, grid)),
                       'spacing': tuple((g[1] - g[0]) for g in grid),
                       'time': None}

        key = 'scalars' if self.value_shape == () else 'vectors'
        print key, self.value_shape
        data = {key: {}}

        if not varname: varname = os.path.basename(base)
        
        for i, t in enumerate(time):
            out = '_'.join([base, str(i)]) + ext

            header_data['time'] = t
            data[key][varname] = self.data[i]

            write_structured_points(out, header_data, data, write_binary)
        

def Coordinate(grid, i):
    '''Grid function for i'th coordinate'''
    assert 0 <= i < len(grid)
    shape = ()  # A scalar
    f = lambda p, grid=grid[i], i=i: grid[p[i]]
    return GenericGridFunction(grid, f, shape)


def SpaceTimeCoordinate(grid):
    '''Vector in space time index -> physical'''
    if len(grid) == 1:
        shape = ()
    else:
        shape = (len(grid), )
    f = lambda p, grid=grid: [grid[i][p[i]] for i in range(len(p))]
    return GenericGridFunction(grid, f, shape)


def TemporalCoordinate(grid):
    '''Time axis'''
    return Coordinate(grid, 0)


def SpatialCoordinate(grid):
    '''Spatial axes'''
    assert len(grid) > 1

    if len(grid) == 2:
        return Coordinate(grid, 1)
    shapee = (len(grid), )
    # Let's try to be more efficient
    y = np.zeros(shape)
    def f(p, grids=grid[1:], y=y):
        for i, pi in enumerate(p[1:]):
            y[i] = grids[i][pi]
        return y
    
    return GenericGridFunction(grid, f, shape)


def diff(f, ntuple, interpolant=Chebyshev, width=7, degree=None):
    '''
    diff(partial derivative) maps a grid function to a grid function of 
    the same shape. Derivative uses polynomial approximation with width 
    points using and a polynomial of give degree.
    '''
    assert isinstance(f, (GenericGridFunction, int, float))
    
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
        # Vectors
        if func.value_shape:
            # Prealoc output
            value_size = np.prod(f.value_shape)
            y = np.zeros((width, value_size))  # Always flat, work
            out = np.zeros(value_size)
            # Compute the n-derivative in i-th direction using degree
            # FIXME: approach that reuse interpolant = derivs up to ?
            def foo(p, y=y, indices=range(value_size), out=out):
                assert isinstance(p, tuple)
                # Eval foo at the line where all by i-th coords are fixed
                # (at their p valeu) and i goes over the stencil
                for y_row, x in zip(y, points(p, i)): np.put(y_row, indices, func(x))
                    
                # Get grid physical point in i direction for interpolation
                Xi = X[i]   
                index_i = p[i]
                x = Xi[stencil(index_i)]  

                # Now we can compute the interpolant of degree and take its
                # derivative at 'p' - it's only the Xi that matters
                p = Xi[index_i]
                out[:] = [interpolant.fit(x, y[:, col], d).deriv(n)(p)
                          for col in range(y.shape[1])]
                # NOTE: we reuse the same pointer for return value.
                # so [foo(p) for in (x, y, z)] will at the end hold
                # 3 values determined by f(z)
                return out
        # Scalar
        else:
            y = np.zeros(width)  # Work array for interpolation

            def foo(p, y=y):
                assert isinstance(p, tuple)
                # Eval foo at the line where all by i-th coords are fixed
                # (at their p valeu) and i goes over the stencil            
                y.ravel()[:] = [func(x) for x in points(p, i)]

                # Get grid physical point in i direction for interpolation
                Xi = X[i]   
                index_i = p[i]
                x = Xi[stencil(index_i)]  

                # Now we can compute the interpolant of degree and take its
                # derivative at 'p' - it's only the Xi that matters
                p = Xi[index_i]
                return interpolant.fit(x, y, d).deriv(n)(p)
        # Still a grid function like
        return GenericGridFunction(func.grid, foo, func.value_shape)

    # NOTE: sorting would allow to use the highest derivative to be
    # computed from the data but it also strongly enforces that the
    # derivatives can be shuffled around.
    # diffs = sorted(diffs, key=lambda p: p[1], reverse=True)
    
    func = f
    while diffs:
        index, power = diffs.pop()
        # At first we diff the function which use the grid data
        # For other directions we use the derivatives
        func = diff_func(func=func, i=index, n=power)
    return func


def grid_from_data(files):
    '''Space time grid for the files'''
    # Orient time axis
    time_axis, files = list(zip(*sorted(zip(map(read_time, files), files), key=lambda p: p[0])))

    # Grid consistency
    space_grid = read_spatial_grid(files[0])
    assert all(space_grid == read_spatial_grid(f) for f in files)

    grid = [list(time_axis)] + space_grid
    return grid


def grid_function_from_data(files, var, split=True):
    '''
    Let each file represent a t-slice. Build a function on the space 
    time cylinder. NOTE: each tensor can be broken into components.
    '''
    grid = grid_from_data(files)
    
    # Consistency of data
    dtype, var = var
    assert dtype in ('vector', 'scalar')

    if dtype == 'scalar':
        extract = lambda f: read_probe_data(f, scalars=(var, ))['scalars'][var]
    else:
        extract = lambda f: read_probe_data(f, vectors=(var, ))['vectors'][var]

    data = map(extract, files)
    shape,  = set(d.shape for d in data)
    # (Npoints, ) or (Npoints, nscalar components)
    data = np.array(data)
    if len(shape) == 1:
        return (GridFunction(grid, data), )

    ncomps = np.prod(shape[1:])        
    data = [data[:, :, i] for i in range(ncomps)]

    if not split:
        return (GridFunction(grid, data), )

    # Scalar components
    return (GridFunction(grid, d) for d in data)
        
# --------------------------------------------------------------------

if __name__ == '__main__':
    import os

    dir = './results'
    files = [os.path.join(dir, f)
             for f in os.listdir(dir) if f.startswith('isine') and 'bin' not in f ]

    f,  = grid_function_from_data(files, var=('scalar', 'foo'))

    pos_x, pos_y, pos_z = grid_function_from_data(files, var=('vector', 'pos.x'))

    f.write('foo_bar.vtk', write_binary=False)

    #pos,  = grid_function_from_data(files, var=('vector', 'pos.x'), split=False)
    #pos.write('foo_bar.vtk', write_binary=False)
