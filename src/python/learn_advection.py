# d u / d t = v*u
from grid_function import GridFunction, Coordinate
from learn_utils import linear_combination, plot_learning_curve
from streams import compile_stream
from system_generation import polynomials, Dx

from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import itertools
import operator


T_end = 10


n_samples = 128
n_train = 200   # Max really

u, wind, x, t = sp.symbols('u w x t')
# Use exact solution to generate the data
u_exact = sp.sin(sp.sin(x) + t)

u_exact = sp.lambdify((t, x), u_exact, 'numpy')

# All data
t_domain = np.linspace(0, T_end, n_samples)
x_domain = np.linspace(-0.5, 0.5, n_samples)
T, X = np.meshgrid(t_domain, x_domain, indexing='ij')
u_data = u_exact(T, X)

# To build the model we sample the data (in interior indices) to get
# the derivative right. For the rhs we keep it simple
u_grid = GridFunction([t_domain, x_domain], u_data)

x_grid = Coordinate([t_domain, x_domain], 1)

lhs_stream = compile_stream((sp.Derivative(u, t), ), {u: u_grid})

foos = polynomials([u], 3)
derivatives = Dx(u, 1, [3])[1:]  # Don't include 0;th
# Combine these guys
columns = [(1./sp.cos(x))*sp.Derivative(u, x)] + \
          list(foos) + \
          list(derivatives) + \
          [reduce(operator.mul, cs) for cs in itertools.product(foos, derivatives)]

rhs_stream = compile_stream(columns, {u:u_grid})

# A subset of this data is now used for training; where we can eval deriv
# safely
train_indices = map(tuple, np.random.choice(np.arange(5, n_samples-5), (n_train, 2)))

# Now build it
lhs = np.array([lhs_stream(point) for point in train_indices])

rhs = np.zeros((len(lhs), len(columns)))
row_indices = np.arange(rhs.shape[1])
for row, point in zip(rhs, train_indices):
    np.put(row, row_indices, rhs_stream(point))

lasso_reg = Lasso(alpha=1E-3)
plot_learning_curve(lasso_reg, rhs, lhs)

plt.show()

print 'd u / dt =', linear_combination(lasso_reg.coef_, columns)
