# We already have a compiler for single expression. For sparse learning
# we want to have many exressions, one make a col entry in the row. However,
# instead of generating a single entry function we could make a function
# for the entire row. Suck a function could you reuse precomputed values.
# E.g. instead of p -> (f(p), f(p)**2) we could make f -> (a -> (a, a**2))(f(p))

import sympy as sp
import itertools
from compiler_utils import *
from sympy_compiler import compile_expr


def extract_primitives(expr):
    '''For the stream these are things that we will compile'''
    # From this point of view a number is not
    if is_number(expr): return ()
    # Symbols become grid functions
    if isinstance(expr, sp.Symbol):
        return (expr, )
    # Derivatives of symbols are compiled into derivatives. We could also
    # make Derivative(f**2, x) a primitive [our compiler can handle this]
    # but a more efficient way is to expand into 2*f*Derivative(f, x)
    if isinstance(expr, sp.Derivative):
        if isinstance(expr.args[0], sp.Symbol):
            return (expr, )
        else:
            return ()
    # Build up the expression from node substitutions
    return sum((extract_primitives(arg) for arg in expr.args), ())


def expand_derivative(expr):
    '''Expand D(f**2, x) into 2*f*D(f, x)'''
    if isinstance(expr, sp.Derivative):
        f, by = expr.args[0], expr.args[1:]
        if f.is_Symbol:
            return expr
        assert is_derivative_free(f), f
        # A chaing rule (D f**2 / D f)(Df / Dx)
        return sum(f.diff(s)*sp.Derivative(s, *by) for s in f.free_symbols)
    # Don't expand where there are no derivatives
    if is_number(expr) or is_derivative_free(expr): return expr
    # Finally build up from nodes
    return type(expr)(*map(expand_derivative, expr.args))


def substitute(expr, match, target):
    '''Replace an expression where math is substituted by taget'''
    # The main reason for not using subs is that it subs Derivative(f, x, y)
    # in Derivative(f, x, y, y)
    if expr == match: return target
    # Not they are different.
    # For a leaf we are done
    if is_number(expr) or not expr.args: return expr
    # Build up node by subbing leaves
    return type(expr)(*(substitute(arg, match, target) for arg in expr.args))


def compile_stream(expr, subs):
    '''
    Given a list of n expression return a function which maps a grid 
    index point to n-tuple where each value corresponds to expression(p)
    '''
    if len(expr) == 1: return compiler_expr(expr, subs)
    # Exapand to reuse derivatives
    expr = map(expand_derivative, expr)
    # Primities are things that will be compiled to grid foo
    primitives = set(sum(map(extract_primitives, expr), ()))
    # Let the derivs be substituted first
    primitives = sorted(primitives, key=lambda k: 1 if k.is_Symbol else 0)

    args, sym_foos = [], []
    count = 0
    # A primitive derivative is substitud by a single function with the
    # goal of 1) transforming expr to things made of of nprimitives symbols
    while primitives:
        primitive = primitives.pop()
        primitive_function = compile_expr(primitive, subs)
        # Subing derivative
        if isinstance(primitive, sp.Derivative):
            # Under a name
            primitive_symbol = sp.Symbol('DD_%d' % count)
            # Now we do substitution to expr
            expr = [substitute(e, primitive, primitive_symbol) for e in expr]
            args.append(primitive_symbol)
            count += 1
        else:
            # No need to sub for symbols
            args.append(primitive)
        sym_foos.append(primitive_function)
    # At this point expr should be defined only in terms of args
    body = sp.lambdify(args, expr)
    # And we're done
    return lambda p, sym_foos=sym_foos: body(*(f(p) for f in sym_foos))

# --------------------------------------------------------------------

if __name__ == '__main__':
    
    from grid_function import GridFunction
    import numpy as np

    n = 32
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

    grid_expr = GridFunction(grid, data)

    grid_expr1 = GridFunction(grid, data+data**2)

    # Check the values at
    i_point = (8, 7, 6, 7)

    # print expand_derivative(sp.Derivative(f**2 + g**2, y, x))

    # exit()
    expression = (1,
                  f,
                  f**2,
                  sp.sin(f)*f,
                  sp.Derivative(f, x),
                  f*sp.Derivative(f, x),
                  f**2 + sp.sqrt(f)*sp.Derivative(f, y),
                  sp.Derivative(f, x, y) + sp.Derivative(g, y, y),
                  sp.Derivative(f**2, y, x) + sp.Derivative(f, y, y))
    subs = {f: grid_expr, g: grid_expr1}
    
    expr_stream = compile_stream(expression, subs)

    import random

    x = list(i_point)
    for _ in range(100):
        random.shuffle(x)
        print x, sum(expr_stream(tuple(x)))

    # FIXME: single ode - learning       (scikit-learn)
    #        several odes - learning
    #        1d pde (u_xx u_x u)
    #        2d pde same
