import sympy as sp
import itertools
from compiler_utils import *
from sympy_compiler import compile_expr


def extract_primitives(expr):
    '''For the stream these are things that we will compile'''
    if isinstance(expr, (sp.Derivative, sp.Symbol)):
        return (expr, )
    return sum((extract_primitives(arg) for arg in expr.args), ())


def substitute(expr, match, target):
    if expr == match: return target
    # Not they are different
    if not expr.args: return expr
    
    return type(expr)(*(substitute(arg, match, target) for arg in expr.args))


def compile_stream(expr, subs):
    if len(expr) == 1: return compiler_expr(expr, subs)

    primitives = set(sum(map(extract_primitives, expr), ()))
    print '\t', primitives
    primitives = sorted(primitives, key=lambda k: 1 if k.is_Symbol else 0)
    print 'primitives', primitives
    args, sym_foos = [], []
    count = 0
    while primitives:
        primitive = primitives.pop()
        primitive_function = compile_expr(primitive, subs)

        if isinstance(primitive, sp.Derivative):
            print primitive, '>>>', expr
            primitive_symbol = sp.Symbol('DD_%d' % count)
            # Now we do substitution to expr
            expr = [substitute(e, primitive, primitive_symbol) for e in expr]
            args.append(primitive_symbol)
            print '<<<', expr
            count += 1
        else:
            args.append(primitive)
            
        sym_foos.append(primitive_function)
    print args
    body = sp.lambdify(args, expr)

    return lambda p, sym_foos=sym_foos: body(*[f(p) for f in sym_foos])

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

    grid_expr = GridFunction(grid, data)
    
    # Check the values at
    i_point = (8, 7, 6, 7)

    
    expression = (f,
                  f**2,
                  sp.sin(f)*f,
                  sp.Derivative(f, x),
                  f*sp.Derivative(f, x),
                  f**2 + sp.sqrt(f)*sp.Derivative(f, y),
                  sp.Derivative(f, x, y),
                  sp.Derivative(f**2, y, x) + sp.Derivative(f, y, y))
    subs = {f: grid_expr}
    
    expr_stream = compile_stream(expression, subs)

    print expr_stream(i_point)
