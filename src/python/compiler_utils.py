import collections, itertools, operator, math
from grid_function import GenericGridFunction
import sympy as sp


# We view numbers as special grid functions (free of grid), and obviously
# scalars

def is_number(n):
    '''A number value symbol '''
    return isinstance(n, (int, float, type(sp.S(3))))


def is_dim_consistent(*grid_fs):
    '''Do grid_fs agree on the space-time dim'''
    return len(set(f.dim for f in grid_fs if not is_number(f))) == 1


def extract_grid(*grid_fs):
    '''Return grid of the functions as long as they agree on the shape'''
    grid = None
    for f in itertools.ifilterfalse(is_number, grid_fs):
        assert grid is None or (len(grid) == len(f.grid) and map(len, grid) == map(len, f.grid))
        grid = f.grid
    return grid


def extract_shape(op, *grid_fs):
    '''
    Grid functions can be combined by +, *, /, ^. Here we chack a shape 
    consistency for the arguments for such operation and return shape of 
    the resulting expression.
    '''
    assert grid_fs
    head, tail = grid_fs[0], grid_fs[1:]
            
    if not tail: return () if is_number(head) else head.value_shape

    first = () if is_number(head) else head.value_shape
    # FIXME: don't allow rank-3 and higher. Their existnce in general.
    # This is simplification until I figure out the operations like *
    assert 0 < len(first) < 3

    if sum(first) == 1: return extract_shape(op, *tail)

    second = extract_shape(op, *tail)
    # NOTE: The first is not number
    if op == '+':        
        assert first == second
        return first
    # NOTE: The first is not number
    if op == '*':
        # Mat/vec vs scalar
        if sum(second) == 1: return first
        # Mat/vec vs vector
        if len(second) == 1:
            # Vector
            if len(first) == 1:
                assert first == second
                return second
            # Mat
            assert first[-1] == second[0]
            # Contract over last axis
            return first[:-1] + second[1:]
        # Mat-mat
        assert first[-1] == second[0]
        # Contract over last axis
        return first[:-1] + second[1:]
   
    # FIXME: Allow these ops only for scalars.
    #        Should *, / be component-wise
    #        Power makes sense for matrices
    assert first == second and len(fist) == 1
    return first


def check_sympy_subs(subs):
    '''Varify map between sympy symbols and grid foos'''
    for sym, grid in subs.iteritems():
        # Scalar
        if isinstance(sym, sp.Symbol):
            assert value_shape == ()
        else:
            assert isinstance(sym, sp.MatrixSymbol)
            assert len(grid.value_shape) > 0
            # Check for possible vectors as COLUMN vectors
            if len(grid.value_shape) == 1:
                nrows, ncols = sym.shape
                assert nrows == grid.value_shape[0] and ncols == 1
            else:
                assert grid.value_shape == sym.shape
    return subs


def gather_exponent(exponent, axis):
    '''Sympy Derivative(f, x, 2, y, 2) to a corresponding tuples'''
    # Get rid of numbers
    def expand(exponent):
        if not exponent: return ()
        if len(exponent) == 1: return exponent

        e0, e1 = exponent[:2]
        
        assert isinstance(e0, sp.Symbol)
        
        if isinstance(e1, int):
            return (e0, )*e1 + expand(exponent[2:])
        else:
            return (e0, ) + expand(exponent[1:])
    exponent = expand(exponent)

    # Now we want to make sure that :
    # 1) we are only using the derivatives in the axis
    assert set(exponent) <= set(axis)

    # And encode the derivative for diffing. Again the orther
    encode = [0]*len(axis)
    for i, a in enumerate(axis):
        if a in exponent:
            degree = exponent.count(a)
            # Must be continuous
            assert exponent[exponent.index(a):exponent.index(a)+degree].count(a) == degree
            encode[i] = degree
    return encode
    

def split(pred, iterable):
    '''Partition iterable into true, false by predicate'''
    t0, t1 = itertools.tee(iterable)
    return filter(pred, t0), [v for v in itertools.ifilterfalse(pred, t1)]


def consume(iterator):
    '''Consume an interator where we don't care about ret val'''
    collections.deque(iterator, maxlen=0)


def is_derivative_free(expr):
    '''Does expression contain derivative'''
    return not isinstance(expr, sp.Derivative) and all(itertools.imap(is_derivative_free, expr.args))


# Translating nodes in the sympy compiler
def apply_add(*fs):
    '''Sum of grid functions'''
    # Consistency check
    assert is_dim_consistent(*fs)
    shape = extract_grid(*fs)
    grid = extract_grid(*fs)

    numbers, grid_fs = split(is_number, fs)
    # + on number only
    if not grid_fs: return sum(numbers)

    assert grid is not None
    nums_sum = sum(numbers)
    op = lambda p, fs=grid_fs, A=nums_sum: A + sum(f(p) for f in fs)
    return GenericGridFunction(grid, op, shape)


def apply_mul(*fs):
    '''Product of grid functions'''
    # Consistency check
    assert is_dim_consistent(*fs)
    shape = extract_grid(*fs)
    grid = extract_grid(*fs)

    numbers, grid_fs = split(is_number, fs)
    # * on number only
    if not grid_fs: return reduce(operator.mul, numbers)

    assert grid is not None
    nums_prod = reduce(operator.mul, numbers) if numbers else 1

    op = lambda p, fs=grid_fs, A=nums_prod: A*reduce(lambda x, y: x*y, (f(p) for f in fs))
    return GenericGridFunction(grid, op, shape)


def apply_pow(foo, power):
    '''Product of grid functions'''
    # Consistency check
    assert is_number(foo) or foo.value_shape == ()  # Scalar only (for now)
    assert is_number(power)

    if is_number(foo): return foo**power

    grid = foo.grid
    value_shape = foo.value_shape

    op = lambda p, f=foo, power=power: (f(p))**power
    return GenericGridFunction(grid, op, value_shape)


def apply_compose(f, g):
    '''Composition f o g'''
    assert is_number(g) or g.value_shape == ()

    # Grab the representation from math (we have scalars)
    f = getattr(math, f)
    if is_number(g): return f(g)

    grid = g.grid
    value_shape = g.value_shape
    op = lambda p, f=f, g=g: f(g(p))
    
    return GenericGridFunction(grid, op, value_shape)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from grid_function import GridFunction
    import numpy as np
    
    f, g = sp.symbols('f g')

    n = 24
    grid = [np.linspace(-1, 4, n),
            np.linspace(0, 2, n),
            np.linspace(0, 1, n),
            np.linspace(-2, 0, n)]
    T, X, Y, Z = np.meshgrid(*grid, indexing='ij')

    f, g, t, x, y, z = sp.symbols('f g t x y z')
    # The one that gives us data
    expr = t**2 + sp.sin(4*x*y*z) + z**2
    
    expr_numpy = sp.lambdify((t, x, y, z), expr, 'numpy')
    data = expr_numpy(T, X, Y, Z)

    grid_f = GridFunction(grid, [[data, 2*data], [3*data, 4*data]])

    #y = np.zeros(grid_f.value_shape)
    print grid_f((8, 7, 6, 7))
    #grid_g = GridFunction(grid, [data, -2*data])

    #print expr_value_shape(f * g, {f: grid_f, g: grid_g})
