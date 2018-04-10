# Here we define operations over grid functions, well any functions of
# one argument, which let us build expressions
from grid_function import diff as diff_grid
from grid_function import GridFunction
import itertools, operator
import sympy as sp


class Coordinate(GridFunction):
    '''Grid function for the i-th coordinate of p[i]'''
    def __init__(self, grid, i):
        assert 0 <= i < len(grid)
        self.grid = grid[i]
        self.i = i

    def __call__(self, p): return np.array([self.grid[p[self.i]]])


# Now I would like to use sympy an create expressions from the grid functions
# This is done by translating nodes in the sympy expression and the idea
# is that the output while not a grid function instance has some of its
# attributes
def grid_foo_like(body, grid):
    body.dim = len(grid)
    body.grid = grid
    return body


def is_number(n): return isinstance(n, (int, float, type(sp.S(3))))

# Translating nodes
def __add(*fs):
    '''A plus'''
    # Consistency check
    dim, grid = None, None
    for f in fs:
        if dim is None:
            dim = f.dim
        assert dim is None or f.dim == dim

        if grid is None:
            grid = f.grid
        # It would be too expensive to check if the grids are identical
        assert grid is None or map(len, grid) == map(len, f.grid)
    # NOTE: consider here grouping the functions and reducing number of calls
    
    op = lambda p, fs=fs: sum(f(p) for f in fs)
    return grid_foo_like(op, grid)


def __prod(*fs):
    '''A mul'''
    # Consistency check
    dim, grid = None, None
    for f in fs:
        if dim is None:
            dim = f.dim
        assert dim is None or f.dim == dim

        if grid is None:
            grid = f.grid
        # It would be too expensive to check if the grids are identical
        assert grid is None or map(len, grid) == map(len, f.grid)
    # NOTE: consider here grouping the functions and reducing number of calls
    
    op = lambda p, fs=fs: reduce(lambda x, y: x*y, (f(p) for f in fs))
    return grid_foo_like(op, grid)


def partition(pred, iterable):
    '''Partition iterable into true, false by predicate'''
    t1, t2 = itertools.tee(iterable)
    return filter(pred, t2), [v for v in itertools.ifilterfalse(pred, t1)]


def compile_from_sympy(expr, subs, grid=None, tx_coords=sp.symbols('t x y z')):
    '''Compiled sympy expression into grid function expression'''
    if grid is None: grid = (subs.values()[0]).grid

    if is_number(expr): return grid_foo_like(lambda p, v=expr: v, grid)

    # Atoms
    if not expr.args:
        # At this point we can only have symbols
        if expr.is_NumberSymbol:
            return compile_from_sympy(expr.n(), subs=subs, grid=grid)
        
        if any(expr == var for var in tx_coords):
            # Maybe it already has been defined
            if expr in subs:
                return subs[expr]
            else:
                # For constructing the coordinate a grid is needed                
                compiled = Coordinate(grid, list(tx_coords).index(expr))
                subs[expr] = compiled
                return compiled

        # Symbol representing function
        assert expr in subs, expr
        return subs[expr]

    # Nodes
    Derivative = type(sp.Derivative(tx_coords[0], tx_coords[1]))
    if isinstance(expr, Derivative):
        # Split into d diffable / d exponents
        diffable = compile_from_sympy(expr.args[0], subs=subs, grid=grid)
        
        exponent = expr.args[1:]
        # We have to be honest and diff only w.r.t to space-time
        assert set(exponent) <= set(tx_coords)
        # Sympy has exponent possible as x x y x. Wishful think for
        # commutativity and group stuff        
        exponent = tuple(expr.args[1:].count(xi) for xi in tx_coords)

        return diff_grid(diffable, exponent)

    # FIXME: New we have a sympy expression which may or may not be free
    # of the derivatives. For the former we rely on sympy; otherwise there
    # is work to be done
    is_derivative = lambda f: isinstance(f, Derivative)
    
    def has_derivative_node(expr):
        return is_derivative(expr) or any(has_derivative_node(arg) for arg in expr.args)
                                    
    if not has_derivative_node(expr):                          
        # Now we rely on sympys lambdify hoping that if does some optimizations
        vars = tuple(expr.free_symbols)

        vars_f = [compile_from_sympy(atom, subs=subs, grid=grid) for atom in vars]
        # The idea is that each of the atoms should only be evaluated once and
        # then the tuple is passed to the lambdified body for evaluation
        body = sp.lambdify(vars, expr, 'numpy')  # FIXME?

        compiled = lambda p, atoms=vars_f, f=body: body(*(atom(p) for atom in atoms))

        return grid_foo_like(compiled, grid=grid)

    # FIXME: treat /, ^, func nodes
    # FIXME: shape information!
    # For a plus node we try to isolate what sympy could lamdify
    if isinstance(expr, type(tx_coords[0] + tx_coords[1])):
        terms, lambdify_pieces = partition(has_derivative_node, expr.args)

        # New sum expression for sympy joins the pile for compilation
        terms.append(reduce(operator.add, lambdify_pieces))
        # Compile piecs
        return __add(*[compile_from_sympy(t, subs=subs, grid=grid) for t in terms])

    # Product node is the same
    if isinstance(expr, type(tx_coords[0] * tx_coords[1])):
        terms, lambdify_pieces = partition(has_derivative_node, expr.args)

        # New sum expression for sympy joins the pile for compilation
        terms.append(reduce(operator.mul, lambdify_pieces))
        # Compile piecs
        return __prod(*[compile_from_sympy(t, subs=subs, grid=grid) for t in terms])

# --------------------------------------------------------------------

if __name__ == '__main__':
    from grid_function import GridFunction
    import numpy as np

    n = 24
    grid = [np.linspace(-1, 4, n),
            np.linspace(0, 2, n),
            np.linspace(0, 1, n),
            np.linspace(-2, 0, n)]
    T, X, Y, Z = np.meshgrid(*grid, indexing='ij')

    f, t, x, y, z = sp.symbols('f t x y z')
    # The one that gives us data
    expr = t**2 + sp.sin(4*x*y*z) + z**2
    
    expr_numpy = sp.lambdify((t, x, y, z), expr, 'numpy')
    data = expr_numpy(T, X, Y, Z)

    grid_expr = GridFunction(grid, [data, -2*data])
    
    # Check the values at
    i_point = (8, 7, 6, 7)
    point = [grid[i][p] for i, p in enumerate(i_point)]
    
    compiled_expr = compile_from_sympy(f**2 + 2*f + x, {f: grid_expr}, grid=grid)
    print compiled_expr(i_point), expr_numpy(*point)

    # # Now take some derivative
    compiled_derivative = compile_from_sympy(sp.Derivative(f, x, y)*f, {f: grid_expr}, grid=grid)
    df_expr = expr*sp.Derivative(expr, x, y)
    # print type(df_expr)
    print compiled_derivative(i_point),
    print sp.lambdify((t, x, y, z), df_expr.doit())(*point)
    
    #print len(Dx(f, [2]))

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
