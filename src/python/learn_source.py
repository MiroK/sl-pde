# Suppose you know the equation and the source?
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


T_end = 5
X_end = 5

n_samples = 128
n_train = 500   # Max really

u, x, t = sp.symbols('u x t')
# Use exact solution to generate the data
u_exact_expr = x**2*t
u_exact = sp.lambdify((t, x), u_exact_expr, 'numpy')

# All data
t_domain = np.linspace(0, T_end, n_samples)
x_domain = np.linspace(0, X_end, n_samples)
T, X = np.meshgrid(t_domain, x_domain, indexing='ij')
u_data = u_exact(T, X)

# To build the model we sample the data (in interior indices) to get
# the derivative right. For the rhs we keep it simple
u_grid = GridFunction([t_domain, x_domain], u_data)

equation = sp.Derivative(u, t) - u - sp.Derivative(u, x) - sp.Derivative(u, x, x)

lhs_stream = compile_stream((equation, ), {u: u_grid})

foos = (-t*x**2, t*x, t, x**2) + (sp.sin(x), sp.cos(x), sp.sin(t), sp.cos(t))
#polynomials([x, t], 3)

# Combine these guys
columns = list(foos)

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

lasso_reg = Lasso(alpha=1E-10)
plot_learning_curve(lasso_reg, rhs, lhs)

plt.show()

print (equation.subs(u, u_exact_expr).doit())

linear_combination(lasso_reg.coef_, columns, sort=True)
