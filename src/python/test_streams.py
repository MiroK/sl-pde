from grid_function import GridFunction, SpaceTimeCoordinate
from streams import compile_stream
import numpy as np
import sympy as sp

n = 32
grid = [np.linspace(-1, 4, n),
        np.linspace(0, 2, n),
        np.linspace(0, 1, n),
        np.linspace(-2, 0, n)]
T, X, Y, Z = np.meshgrid(*grid, indexing='ij')

f, g, t, x, y, z = sp.symbols('f g t x y z')

expressions = (x, #t**2 + sp.sin(4*x*y*z) + z**2,
               y)#t + sp.exp(-(4*(x**2 + y**2))) + z**3)

grid_fs = []
for f_ in expressions:
    expr_numpy = sp.lambdify((t, x, y, z), f_, 'numpy')
    data = expr_numpy(T, X, Y, Z)
    grid_fs.append(GridFunction(grid, data))

# Not the streams
expression = (1,
              f,
              g,
              f**2 + f*g + g**2, sp.sin(f+g), sp.exp(-(f**2 + g**2)),
              sp.Derivative(f, x),
              sp.Derivative(f**2, y),
              sp.Derivative(f+g, x),
              2*f*sp.Derivative(g, y, x),
              2*g*sp.Derivative(f, y, x),
              sp.Derivative(2*f*g, y, x, y),
              sp.Derivative(f**2, x, x)
)

from streams import expand_derivative
gg = expand_derivative(expression[0])

print 
subs = {f: grid_fs[0], g: grid_fs[1]}

# Sympy version
expr_stream = compile_stream(expression, subs)

f, g = expressions
expression = (1,
              f,
              g,
              f**2 + f*g + g**2, sp.sin(f+g), sp.exp(-(f**2 + g**2)),
              f.diff(x, 1), (f**2).diff(y), (f+g).diff(x),
              2*f*(g.diff(y, x)),
              2*g*(f.diff(y, x)),
              (2*f*g).diff(x, y, y),
              (f**2).diff(x, x)
)

sympy_stream = sp.lambdify((t, x, y, z), expression)

# Check the values at
i_point = (8, 7, 6, 7)

i_to_x = SpaceTimeCoordinate(grid)
x = i_to_x(i_point)

approx = expr_stream(i_point)
exact = sympy_stream(*x)

for a, e in zip(approx, exact):
    print abs(a - e), a, e
