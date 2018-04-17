# Learn a simple second oreder ODE d^2 u / dt^2 = -u
from grid_function import GridFunction
from learn_utils import linear_combination, plot_learning_curve
from streams import compile_stream
from system_generation import polynomials

from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import itertools
import operator


T_end = 5
n_samples = 1000
n_train = 100   # Max really

u, t = sp.symbols('u t')
# Use exact solution to generate the data
u_exact = sp.lambdify(t, sp.sin(t), 'numpy')

# All data
t_domain = np.linspace(0, T_end, n_samples)
u_data = u_exact(t_domain)

# To build the model we sample the data (in interior indices) to get
# the derivative right. For the rhs we keep it simple
u_grid = GridFunction([t_domain], u_data)
lhs_stream = compile_stream((sp.Derivative(u, t, 2), ), {u: u_grid})

poly = polynomials([u], 4)
foos = (sp.sin(u), sp.cos(u))

columns = list(poly) + list(foos) + [reduce(operator.mul, cs) for cs in itertools.product(poly, foos)]

rhs_stream = compile_stream(columns, {u:u_grid})

# A subset of this data is now used for training; where we can eval deriv
# safely
train_indices = np.random.choice(np.arange(5, n_samples-5), n_train)

# Now build it
lhs = np.array([lhs_stream((point, )) for point in train_indices])

rhs = np.zeros((len(lhs), len(columns)))
row_indices = np.arange(rhs.shape[1])
for row, point in zip(rhs, train_indices):
    np.put(row, row_indices, rhs_stream((point,)))

lasso_reg = Lasso(alpha=0.01)
plot_learning_curve(lasso_reg, rhs, lhs)

plt.show()

print 'd^2 u / dt^2 =', linear_combination(lasso_reg.coef_, columns)
