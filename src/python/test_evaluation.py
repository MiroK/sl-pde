from evaluation import GridFunction, diff
from itertools import izip
import numpy as np
import sympy as sp

TOL = 1E-10

def test_1d_poly(n):
    '''Error for some approx at [-1, 4]'''
    t = sp.Symbol('t')
    t_domain = np.linspace(-1, 4, n)  # Physical
    t_domain_index = range(len(t_domain))

    # I will look at 3 points aroung the center
    pts_index = [t_domain_index[len(t_domain)/2-1],
                 t_domain_index[len(t_domain)/2],
                 t_domain_index[len(t_domain)/2+1]]
    pts = t_domain[pts_index]

    symbols = (t, )
    functions = [2*t, 3*(t**2-2*t+1), (t**4 - 2*t**2 + t - 1)]
    # We look at up to 3 derivatives
    derivs = [(0, ), (1, ), (2, ), (3, )]
    status = True
    for foo in functions:
        print foo
        foo_values = np.array(map(sp.lambdify(t, foo), t_domain))
        grid_foo = GridFunction([t_domain], foo_values)
        for d in derivs:
            if d != (0, ):
                dfoo = foo
                for var, deg in zip(symbols, d): dfoo = dfoo.diff(var, deg)
            else:
                dfoo = foo
            print '\t', d, dfoo,
            # True
            df = sp.lambdify(t, dfoo)
            exact = df(pts)

            dgrid_foo = diff(grid_foo, d)
            numeric = np.array([dgrid_foo([i]) for i in pts_index]).flatten()

            error = exact - numeric
            e = np.linalg.norm(error)
            print 'OK' if e < TOL else (exact, numeric)

            status = status and e < TOL
    return status


def test_1d_approx(n, width=7):
    '''Error for some approx at [-1, 4]. Just convergence'''
    t = sp.Symbol('t')
    t_domain = np.linspace(-1, 4, n)  # Physical
    t_domain_index = range(len(t_domain))

    # I will look at 3 points aroung the center
    pts_index = [t_domain_index[len(t_domain)/2-1],
                 t_domain_index[len(t_domain)/2],
                 t_domain_index[len(t_domain)/2+1]]
    pts = t_domain[pts_index]

    symbols = (t, )
    functions = [2*sp.sin(sp.pi*t), sp.cos(2*sp.pi*t**2)]
    # We look at up to 3 derivatives
    derivs = [(1, ), (2, ), (3, ), (4, )]
    
    status = {f: list() for f in functions}
    for foo in functions:
        foo_values = np.array(map(sp.lambdify(t, foo), t_domain))
        grid_foo = GridFunction([t_domain], foo_values)
        for d in derivs:
            if d != (0, ):
                dfoo = foo
                for var, deg in zip(symbols, d): dfoo = dfoo.diff(var, deg)
            else:
                dfoo = foo
            # True
            df = sp.lambdify(t, dfoo)
            exact = np.array(map(df, pts))

            dgrid_foo = diff(grid_foo, d, width=width)
            numeric = np.array([dgrid_foo([i]) for i in pts_index]).flatten()

            error = exact - numeric
            e = np.linalg.norm(error)
            status[foo].append(e)
            
    return status


def check_approx(gen, n_values):
    '''Consume gen over n_values returning stats for each key'''
    n_values = np.array(map(float, n_values))
    # Make room for the next ns 
    data = gen(n_values[0])
    for f in data: data[f] = [[e] for e in data[f]]
    
    for n in n_values[1:]:
        # Append errors of derivatives for other n
        for f, errors in gen(n).iteritems():
            [data[f][i].append(error) for i, error in enumerate(errors)]

    dn = np.log(n_values[1:]/n_values[:-1])
    # Statistics for individual functions
    status = True
    for f, errors in data.iteritems():
        print '============= %s ============' % f
        streams = []
        for error in map(np.array, errors):
            rates = -np.r_[np.nan, np.log(error[1:]/error[:-1])/dn]
            # Convergence in all derivatives
            status = status and all(r > 0 or np.isnan(r) for r in rates)
            streams.append(
                (lambda es=error, rs=rates: ('%.4E[%.2f]' % er for er in zip(es, rs)))()
            )

        for row in izip(*streams):
            print row

    return status
# --------------------------------------------------------------------

if __name__ == '__main__':
    # Polynomial must be exact
    assert test_1d_poly(10)

    # With other functions there should be some convergence
    n_values = (32, 64, 128, 256)
    assert check_approx(test_1d_approx, n_values)

    exit()
    ## Dim 1
    t = np.linspace(0, 4, 15)

    f = lambda t: t
    U = f(t)

    gf = GridFunction([t], U)

    exit()

    t = np.linspace(0, 1, 15)
    x = np.linspace(1, 2, 15)
    y = np.linspace(-1, 1, 15) 

    T, X, Y = np.meshgrid(t, x, y, indexing='ij')

    f = lambda (t, x, y): 2*t + 3*x + 4*y*x**3
    U = f((T, X, Y))

    gf = GridFunction([t, x, y], U)

    p = (1, 2, 3)
    print diff(gf, (0, 3, 1))(p)#, f((t[p[0]], x[p[1]], y[p[2]]))
