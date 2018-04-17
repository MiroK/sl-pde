# We break d^2 u / dt^2 = -u
# as d/dt[u, v] = [0  1;[u;
#                  -1 0] v]
#
from grid_function import GridFunction
from learn_utils import linear_combination, plot_learning_curve
from system_generation import polynomials
from streams import compile_stream

from sklearn.linear_model import Lasso
from scipy.sparse import bmat
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

n_eqs = 2

T_end = 8
n_samples = 1000
n_train = 100   # Max really

t = sp.Symbol('t')
# This is now a vector valued function
u_exact = lambda t: np.array([np.sin(t), np.cos(t)])

# All data
t_domain = np.linspace(0, T_end, n_samples)
u_data = np.array(map(u_exact, t_domain))

# Let's make one grid function for evvery component of that vector
u_grids = []
for col in range(n_eqs):
    u_grids.append(GridFunction([t_domain], u_data[:, col]))
# And a symbols for each
us = sp.symbols(' '.join(['u%d' % i for i in range(n_eqs)]))

symbol_foo_map = dict(zip(us, u_grids))
# We are going to build a block structured system d u0/dt @ all_pts, next component
lhs_streams = [compile_stream((sp.Derivative(usi, t), ), symbol_foo_map) for usi in us]

# NOTE: this is reused for each block
columns = polynomials(us, 2)
print columns

rhs_stream = compile_stream(columns, symbol_foo_map)

# A subset of this data is now used for training; where we can eval deriv
# safely
train_indices = np.random.choice(np.arange(5, n_samples-5), n_train)

# Now build it
lhs = np.hstack([[lhs_stream((point, )) for point in train_indices]
                   for lhs_stream in lhs_streams])
# This is a block
rhs = np.zeros((len(lhs)/n_eqs, len(columns)))
row_indices = np.arange(rhs.shape[1])
for row, point in zip(rhs, train_indices):
    np.put(row, row_indices, rhs_stream((point,)))
# The matrix for rhs is 'block' diagonal
rhs = bmat([[rhs if i == j else None for i in range(n_eqs)] for j in range(n_eqs)])
rhs = rhs.todense()

lasso_reg = Lasso(alpha=2.5E-5)

plot_learning_curve(lasso_reg, rhs, lhs)
plt.show()

# lasso_reg.fit(rhs, lhs)

print lasso_reg.coef_
for eq in range(n_eqs):
    coef = lasso_reg.coef_[np.arange(eq*len(columns), (eq+1)*len(columns))]
    print 'd u_{%d} / dt =' % eq, linear_combination(coef, columns)

# FIXME: scale features
# TODO: extend to comples - then we could have systems which conserve energy
# Train on some data set generated by foo, then use other .... (restart)
# NOTE: what we'd really like to do here is to also add symmetry constraints
# 
# Priority is to have a 1d/2d PDE to 'impress' them
#
#
