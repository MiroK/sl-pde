# We compile sympy expressions to GenericGridFunctions

from grid_function import Coordinate, GenericGridFunction
from grid_function import diff as diff_grid
from compiler_utils import *
import itertools, operator
import sympy as sp


def compile_from_expr(expr, subs, grid=None, tx_coords=sp.symbols('t x y z')):
    '''Compiled sympy expression into grid function expression'''
    if grid is None: grid = (subs.values()[0]).grid

    # A grid function that is number
    if is_number(expr):
        if isinstance(expr, (int, float)):
            return expr
        # Eval and ask again
        return compile_from_expr((int if expr.is_Inger else float)(expr), subs, grid)

    # Atoms
    if not expr.args:
        # At this point we can only have not numberic symbols 
        # Coordinate symbols
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
    if isinstance(expr, sp.Derivative):
        # Split into d diffable / d exponents
        # Compule the symbolic body
        diffable = compile_from_expr(expr.args[0], subs=subs, grid=grid)
        print 'diffable', expr
        exponent = expr.args[1:]
        # We have to be honest and diff only w.r.t to space-time
        assert set(exponent) <= set(tx_coords)
        # Sympy has exponent possible as x x y x. Wishful think for
        # commutativity and group stuff, i.e. the same assumption as in
        # diff_grid
        exponent = tuple(expr.args[1:].count(xi) for xi in tx_coords)

        return diff_grid(diffable, exponent)

    # FIXME: New we have a sympy expression which may or may not be free
    # of the derivatives. For the former we rely on sympy; otherwise there
    # is work to be done
    print expr, is_derivative_free(expr)
    if is_derivative_free(expr):
        print 'Applies', expr
        # Now we rely on sympys lambdify hoping that if does some optimizations
        vars = tuple(expr.free_symbols)
        # FIXME: However, for now the code gen should only use scalars
        assert all(v.is_Symbol for v in vars)
        # FIXME: Now verify the shape of sympy to be scalar ...

        vars_f = [compile_from_expr(atom, subs=subs, grid=grid) for atom in vars]
        # The idea is that each of the atoms should only be evaluated once and
        # then the tuple is passed to the lambdified body for evaluation
        body = sp.lambdify(vars, expr, 'math')  # Or numpy? 

        work = np.zeros(len(vars_f))
        def compiled(p, f=body, atoms=vars_f, work=work, indices=range(len(work))):
            # Evaluate atoms once
            np.put(work, indices, [atom(p) for atom in atoms])
            # Use in body
            return f(*work)

        return GenericGridFunction(grid, compiled)

    # For a plus node we try to isolate what sympy could lamdify
    if isinstance(expr, sp.Add):
        terms, lambdify_pieces = split(is_derivative_free, expr.args)
        # New sum expression for sympy joins the pile for compilation
        terms.append(reduce(operator.add, lambdify_pieces))
        # Into grid functions
        return apply_add(*[compile_from_expr(t, subs=subs, grid=grid) for t in terms])

    if isinstance(expr, sp.Mul):
        terms, lambdify_pieces = split(is_derivative_free, expr.args)
        # New sum expression for sympy joins the pile for compilation
        terms.append(reduce(operator.mul, lambdify_pieces))
        # Into grid functions
        return apply_mul(*[compile_from_expr(t, subs=subs, grid=grid) for t in terms])
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from grid_function import GridFunction
    import numpy as np

    values = np.zeros(1)
    
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

    grid_expr = GridFunction(grid, data)
    
    # Check the values at
    i_point = (8, 7, 6, 7)
    point = [grid[i][p] for i, p in enumerate(i_point)]
    
    compiled_expr = compile_from_expr(f**3 + f*sp.Derivative(f, x, y)**2, {f: grid_expr}, grid=grid)
    values[:] = compiled_expr(i_point)

    print values, 

    # # Now take some derivative
    df_expr = expr**2 + expr*sp.Derivative(expr, x, y)

    print sp.lambdify((t, x, y, z), df_expr.doit())(*point)
