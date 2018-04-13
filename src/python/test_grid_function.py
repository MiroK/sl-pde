from grid_function import GridFunction, diff
from collections import defaultdict
from itertools import izip
import numpy as np
import sympy as sp

TOL = 1E-10

# 1-4d polynomials ---------------------------------------------------
# Want exactness

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

    numeric = np.zeros(len(pts))
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
            numeric[:] = [dgrid_foo((pi, )) for pi in pts_index]

            error = exact - numeric
            e = np.linalg.norm(error)
            print 'OK' if e < TOL else (exact, numeric)

            status = status and e < TOL
    return status


def test_1d_poly_vec(n):
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

    numeric = np.zeros((len(pts), 2))
    for foo in functions:
        print foo
        foo_values = np.array(map(sp.lambdify(t, foo), t_domain))
        grid_foo = GridFunction([t_domain], [foo_values, 2*foo_values])
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
            exact = np.c_[exact, 2*exact]

            dgrid_foo = diff(grid_foo, d)

            for row, pi in zip(numeric, pts_index):
                np.put(row, range(2), dgrid_foo((pi, )))

            error = exact - numeric
            e = np.linalg.norm(error)
            print 'OK' if e < TOL else (exact, numeric)

            status = status and e < TOL
    return status


def test_2d_poly(n):
    '''Error for some approx at [-1, 4]x[0, 2]'''
    t, x = sp.symbols('t x')
    
    t_domain = np.linspace(-1, 4, n)  # Physical
    x_domain = np.linspace(0, 2, n)   # Physical
    T, X = np.meshgrid(t_domain, x_domain, indexing='ij')

    # I will look at 3 points aroung the center
    pts_indices = [(n/2-1, )*2, (n/2, )*2, (n/2+1, )*2]
    # Their physical representations
    pts = map(lambda (i, j): [t_domain[i], x_domain[j]], pts_indices)

    symbols = (t, x)
    functions = [2*t+x,
                 3*(t**2-2*t+1) + x**2,
                 (t**4 - 2*t**2 + t - 1) + 2*t**2*4*x**3 - x*3*t**2]
    # Some deriveatives
    derivs = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 3), (2, 2)]

    numeric = np.zeros(len(pts))    
    status = True
    for foo in functions:
        print foo
        foo_values = sp.lambdify((t, x), foo, 'numpy')(T, X)
        grid_foo = GridFunction([t_domain, x_domain], foo_values)

        for d in derivs:
            if d != (0, 0):
                dfoo = foo
                for var, deg in zip(symbols, d): dfoo = dfoo.diff(var, deg)
            else:
                dfoo = foo
            print '\t', d, dfoo,
            # True
            df = sp.lambdify((t, x), dfoo)
            exact = np.array([df(*pt) for pt in pts])

            dgrid_foo = diff(grid_foo, d)
            [dgrid_foo(pi, numeric[i:i+1]) for i, pi in enumerate(pts_indices)]

            error = exact - numeric
            e = np.linalg.norm(error)
            print 'OK' if e < TOL else (exact, numeric)
            status = status and e < TOL
    return status


def test_3d_poly(n):
    t, x, y = sp.symbols('t x y')
    
    t_domain = np.linspace(-1, 4, n)  # Physical
    x_domain = np.linspace(0, 2, n)   
    y_domain = np.linspace(0, 1, n)
    T, X, Y = np.meshgrid(t_domain, x_domain, y_domain, indexing='ij')

    # I will look at 3 points aroung the center
    pts_indices = [(n/2-1, )*3, (n/2, )*3, (n/2+1, )*3]
    # Their physical representations
    pts = map(lambda (i, j, k): [t_domain[i], x_domain[j], y_domain[k]],
              pts_indices)

    symbols = (t, x, y)
    functions = [2*t+1*x-2*y,
                 3*(t**2-2*t+1) + x**2 + t*x - 4*t*y + 3*t*x,
                 y*(t**4 - 2*t**2 + t - 1) + 2*t**2*4*x**3 - y*x*4*t**4]
    # Some deriveatives
    derivs = [(0, 0, 0), (1, 0, 1), (0, 1, 0), (2, 1, 0), (1, 2, 0), (3, 1, 1),
              (1, 3, 1)]

    status = True
    numeric = np.zeros(len(pts))
    for foo in functions:
        print foo
        foo_values = sp.lambdify((t, x, y), foo, 'numpy')(T, X, Y)
        grid_foo = GridFunction([t_domain, x_domain, y_domain], foo_values)

        for d in derivs:
            if d != (0, 0, 0):
                dfoo = foo
                for var, deg in zip(symbols, d): dfoo = dfoo.diff(var, deg)
            else:
                dfoo = foo
            print '\t', d, dfoo,
            # True
            df = sp.lambdify((t, x, y), dfoo)
            exact = np.array([df(*pt) for pt in pts])

            dgrid_foo = diff(grid_foo, d)
            [dgrid_foo(pi, numeric[i:i+1]) for i, pi in enumerate(pts_indices)]

            error = exact - numeric
            e = np.linalg.norm(error)
            print 'OK' if e < TOL else (exact, numeric)
            status = status and e < TOL
    return status


def test_4d_poly(n):
    t, x, y, z = sp.symbols('t x y z')
    
    t_domain = np.linspace(-1, 4, n)  # Physical
    x_domain = np.linspace(0, 2, n)   
    y_domain = np.linspace(0, 1, n)
    z_domain = np.linspace(-2, 0, n)
    T, X, Y, Z = np.meshgrid(t_domain, x_domain, y_domain, z_domain, indexing='ij')

    # I will look at 3 points aroung the center
    pts_indices = [(n/2-1, )*4, (n/2, )*4, (n/2+1, )*4]
    # Their physical representations
    pts = map(lambda (i, j, k, l): [t_domain[i], x_domain[j], y_domain[k], z_domain[k]],
              pts_indices)

    symbols = (t, x, y, z)
    functions = [2*t+1*x**3-2*y**4+z**2,
                 3*(t**2-2*t+1) + z**2*x**2 + t*x**4 - 4*t*y*z**2*x + 3*t*x]
    # Some deriveatives
    derivs = [(0, 0, 3, 0), (1, 0, 1, 1), (0, 1, 0, 2), (2, 1, 0, 3), (1, 1, 2, 1)]

    numeric = np.zeros(len(pts))
    status = True
    for foo in functions:
        print foo
        foo_values = sp.lambdify((t, x, y, z), foo, 'numpy')(T, X, Y, Z)
        grid_foo = GridFunction([t_domain, x_domain, y_domain, z_domain], foo_values)

        for d in derivs:
            if d != (0, 0, 0, 0):
                dfoo = foo
                for var, deg in zip(symbols, d): dfoo = dfoo.diff(var, deg)
            else:
                dfoo = foo
            print '\t', d, dfoo,
            # True
            df = sp.lambdify((t, x, y, z), dfoo)
            exact = np.array([df(*pt) for pt in pts])

            dgrid_foo = diff(grid_foo, d)
            [dgrid_foo(pi, numeric[i:i+1]) for i, pi in enumerate(pts_indices)]

            error = exact - numeric
            e = np.linalg.norm(error)
            print 'OK' if e < TOL else (exact, numeric)
            status = status and e < TOL
    return status

# 1-4d functions -----------------------------------------------------
# Some convergence

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

    numeric = np.zeros(len(pts))
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
            numeric.ravel()[:] = [dgrid_foo((pi, )) for pi in pts_index]

            error = exact - numeric
            e = np.linalg.norm(error)
            status[foo].append(e)
            
    return status


def test_2d_approx(n):
    '''Error for some approx at [-1, 4]x[0, 2].Just convergence'''
    t, x = sp.symbols('t x')
    
    t_domain = np.linspace(-1, 4, n)  # Physical
    x_domain = np.linspace(0, 2, n)   # Physical
    T, X = np.meshgrid(t_domain, x_domain, indexing='ij')

    # I will look at 3 points aroung the center
    pts_indices = [(n/2-1, )*2, (n/2, )*2, (n/2+1, )*2]
    # Their physical representations
    pts = map(lambda (i, j): [t_domain[i], x_domain[j]], pts_indices)

    symbols = (t, x)
    functions = [sp.sin(sp.pi*(t+x)),
                 t*sp.sin(2*sp.pi*(x -3*t*x)),
                 sp.cos(2*sp.pi*x*t)]

    # Some deriveatives
    derivs = [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 3), (2, 2)]
    errors = {foo: list() for foo in functions}
    numeric = np.zeros(len(pts))
    for foo in functions:

        foo_values = sp.lambdify((t, x), foo, 'numpy')(T, X)
        grid_foo = GridFunction([t_domain, x_domain], foo_values)

        for d in derivs:
            if d != (0, 0):
                dfoo = foo
                for var, deg in zip(symbols, d): dfoo = dfoo.diff(var, deg)
            else:
                dfoo = foo

            # True
            df = sp.lambdify((t, x), dfoo)
            exact = np.array([df(*pt) for pt in pts])

            dgrid_foo = diff(grid_foo, d)
            [dgrid_foo(pi, numeric[i:i+1]) for i, pi in enumerate(pts_indices)]

            error = exact - numeric
            e = np.linalg.norm(error)
            errors[foo].append(e)
    return errors


def test_3d_approx(n):
    t, x, y = sp.symbols('t x, y')
    
    t_domain = np.linspace(-1, 4, n)  # Physical
    x_domain = np.linspace(0, 2, n)   
    y_domain = np.linspace(0, 1, n)
    T, X, Y = np.meshgrid(t_domain, x_domain, y_domain, indexing='ij')

    # I will look at 3 points aroung the center
    pts_indices = [(n/2-1, )*3, (n/2, )*3, (n/2+1, )*3]
    # Their physical representations
    pts = map(lambda (i, j, k): [t_domain[i], x_domain[j], y_domain[k]],
              pts_indices)

    symbols = (t, x, y)
    functions = [sp.sin(sp.pi*t)*sp.exp(-(x**2 + y**2)), 
                 sp.cos(sp.pi*t)*sp.exp(-(x**2 + y**2))]
    # Some deriveatives
    derivs = [(1, 0, 1), (2, 1, 1), (3, 1, 2), (1, 3, 1)]

    errors = defaultdict(list)
    numeric = np.zeros(len(pts))
    for foo in functions:
        foo_values = sp.lambdify((t, x, y), foo, 'numpy')(T, X, Y)
        grid_foo = GridFunction([t_domain, x_domain, y_domain], foo_values)

        for d in derivs:
            if d != (0, 0, 0):
                dfoo = foo
                for var, deg in zip(symbols, d): dfoo = dfoo.diff(var, deg)
            else:
                dfoo = foo

            # True
            df = sp.lambdify((t, x, y), dfoo)
            exact = np.array([df(*pt) for pt in pts])

            dgrid_foo = diff(grid_foo, d)
            [dgrid_foo(pi, numeric[i:i+1]) for i, pi in enumerate(pts_indices)]

            error = exact - numeric
            e = np.linalg.norm(error)
            errors[foo].append(e)
    return errors


def test_4d_approx(n):
    t, x, y, z = sp.symbols('t x y z')
    
    t_domain = np.linspace(-1, 4, n)  # Physical
    x_domain = np.linspace(0, 2, n)   
    y_domain = np.linspace(0, 1, n)
    z_domain = np.linspace(-2, 0, n)
    T, X, Y, Z = np.meshgrid(t_domain, x_domain, y_domain, z_domain, indexing='ij')

    # I will look at 3 points aroung the center
    pts_indices = [(n/2-1, )*4, (n/2, )*4, (n/2+1, )*4]
    # Their physical representations
    pts = map(lambda (i, j, k, l): [t_domain[i], x_domain[j], y_domain[k], z_domain[k]],
              pts_indices)

    symbols = (t, x, y, z)

    functions = [sp.sin(sp.pi*t)*sp.exp(-(x**2 + y**3 - z**2)), 
                 sp.cos(sp.pi*t)*sp.exp(-(x**2 + 3*y**2 + x*y*z))]
                
    # Some deriveatives
    derivs = [(0, 0, 3, 0), (1, 0, 1, 1), (0, 1, 0, 2), (2, 1, 0, 3), (1, 1, 2, 1)]
    numeric = np.zeros(len(pts))
    errors = defaultdict(list)
    for foo in functions:
        foo_values = sp.lambdify((t, x, y, z), foo, 'numpy')(T, X, Y, Z)
        grid_foo = GridFunction([t_domain, x_domain, y_domain, z_domain], foo_values)

        for d in derivs:
            if d != (0, 0, 0, 0):
                dfoo = foo
                for var, deg in zip(symbols, d): dfoo = dfoo.diff(var, deg)
            else:
                dfoo = foo

            # True
            df = sp.lambdify((t, x, y, z), dfoo)
            exact = np.array([df(*pt) for pt in pts])

            dgrid_foo = diff(grid_foo, d)
            [dgrid_foo(pi, numeric[i:i+1]) for i, pi in enumerate(pts_indices)]

            error = exact - numeric
            e = np.linalg.norm(error)
            errors[foo].append(e)
    return errors

# ----

def check_approx(gen, n_values):
    '''Consume gen over n_values returning stats for each key'''
    n_values = np.array(n_values)
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
        # Print convergence
        for row in izip(*streams): print row
        
    return status

# --------------------------------------------------------------------

if __name__ == '__main__':
    # Checks for dimensions
    checks = [True, 0, 0, 0]#True, True, True]
    
    # t
    if checks[0]:
        # Polynomial must be exact
        assert test_1d_poly(10)
        # Also for tensor valued
        assert test_1d_poly_vec(10)

        # With other functions there should be some convergence
        n_values = (32, 64, 128, 256)
        assert check_approx(test_1d_approx, n_values)
    # t x 
    if checks[1]:
        # Polynomial must be exact
        test_2d_poly(10)

        # With other functions there should be some convergence
        n_values = (32, 64, 128)
        assert check_approx(test_2d_approx, n_values)
        
    # FIXME: update below
    # t x y
    if checks[2]:
        # Polynomial must be exact
        test_3d_poly(10)

        # With other functions there should be some convergence
        n_values = (32, 64, 128)
        assert check_approx(test_3d_approx, n_values)
    # t x x z
    if checks[3]:
        # Polynomial must be exact
        test_4d_poly(10)

        # With other functions there should be some convergence
        n_values = (24, 48, 96)
        assert check_approx(test_4d_approx, n_values)


    # exit()
    # ## Dim 1
    # t = np.linspace(0, 4, 15)

    # f = lambda t: t
    # U = f(t)

    # gf = GridFunction([t], U)

    # exit()

    # t = np.linspace(0, 1, 15)
    # x = np.linspace(1, 2, 15)
    # y = np.linspace(-1, 1, 15) 

    # T, X, Y = np.meshgrid(t, x, y, indexing='ij')

    # f = lambda (t, x, y): 2*t + 3*x + 4*y*x**3
    # U = f((T, X, Y))

    # gf = GridFunction([t, x, y], U)

    # p = (1, 2, 3)
    # print diff(gf, (0, 3, 1))(p)#, f((t[p[0]], x[p[1]], y[p[2]]))
