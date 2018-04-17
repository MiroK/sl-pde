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

n_samples = 128
n_train = 200   # Max really

u, x, t = sp.symbols('u x t')
# Use exact solution to generate the data
u_exact = sum(sp.exp(-(k**2)*t)*sp.sin(k*x) for k in range(1, 4))#, 'numpy')
assert sp.simplify(u_exact.diff(t, 1) - u_exact.diff(x, 2)) == 0

u_exact = sp.lambdify((t, x), u_exact, 'numpy')

# All data
t_domain = np.linspace(0, T_end, n_samples)
x_domain = np.linspace(0, X_end, n_samples)
T, X = np.meshgrid(t_domain, x_domain, indexing='ij')
u_data = u_exact(T, X)

# To build the model we sample the data (in interior indices) to get
# the derivative right. For the rhs we keep it simple
u_grid = GridFunction([t_domain, x_domain], u_data)
lhs_stream = compile_stream((sp.Derivative(u, t), ), {u: u_grid})

foos = polynomials([u], 3)
derivatives = Dx(u, 1, [3])[1:]  # Don't include 0;th
# Combine these guys
columns = list(foos) + \
          list(derivatives) + \
          [reduce(operator.mul, cs) for cs in itertools.product(foos, derivatives)]


rhs_stream = compile_stream(columns, {u:u_grid})

# A subset of this data is now used for training; where we can eval deriv
# safely
train_indices = map(tuple, np.random.choice(np.arange(5, n_samples-5), (n_train, 2)))
print len(train_indices)

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
