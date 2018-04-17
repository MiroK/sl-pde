# Learn a simple ODE d u / dt = u
from grid_function import GridFunction
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
X_end = np.pi
Y_end = 2*np.pi

n_samples = 128
n_train = 200   # Max really

u, x, y, t = sp.symbols('u x y t')
# Use exact solution to generate the data
u_exact = sum(sp.exp(-(k**2+l**2)*t)*sp.sin(k*x)*sp.sin(l*y)
              for k in range(1, 4) for l in range(2, 5))
assert sp.simplify(u_exact.diff(t, 1) - u_exact.diff(x, 2) - u_exact.diff(y, 2)) == 0

u_exact = sp.lambdify((t, x, y), u_exact, 'numpy')

# All data
t_domain = np.linspace(0, T_end, n_samples)
x_domain = np.linspace(0, X_end, n_samples)
y_domain = np.linspace(0, Y_end, n_samples)

T, X, Y = np.meshgrid(t_domain, x_domain, y_domain, indexing='ij')
u_data = u_exact(T, X, Y)

# To build the model we sample the data (in interior indices) to get
# the derivative right. For the rhs we keep it simple
u_grid = GridFunction([t_domain, x_domain, y_domain], u_data)
lhs_stream = compile_stream((sp.Derivative(u, t), ), {u: u_grid})

foos = polynomials([u], 3)
derivatives = Dx(u, 2, [3])[1:]  # Don't include 0;th
# Combine these guys
columns = list(foos) + \
          [sp.Derivative(u, x, x) + sp.Derivative(u, y, y)] +\
          list(derivatives) + \
          [reduce(operator.mul, cs) for cs in itertools.product(foos, derivatives)]

print columns

rhs_stream = compile_stream(columns, {u:u_grid})

# A subset of this data is now used for training; where we can eval deriv
# safely
train_indices = map(tuple, np.random.choice(np.arange(5, n_samples-5), (n_train, u_grid.dim)))
print len(train_indices)

# Now build it
lhs = np.array([lhs_stream(point) for point in train_indices])

rhs = np.zeros((len(lhs), len(columns)))
row_indices = np.arange(rhs.shape[1])
for row, point in zip(rhs, train_indices):
    np.put(row, row_indices, rhs_stream(point))

lasso_reg = Lasso(alpha=2E-5)
plot_learning_curve(lasso_reg, rhs, lhs)

plt.show()

print 'd u / dt =', linear_combination(lasso_reg.coef_, columns)
