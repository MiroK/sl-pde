from probe_io import read_probe_data, read_time, read_spatial_grid
import numpy as np
import itertools

# The idead here is that on c side with have generated data on each
# cpu - we have many files. Then we perform merging to end up with
# one file per time step. Here we check that after the merge the data
# is correct

# Data obtained by interpolation (the function was linear in x, y, z)
# should be exact; how they were generated
f = lambda (t, (x, y, z)): x + y*t + z

path = './results/isine_%d_0.vtk'

failed = False
max_error = 0
for i in range(15):
    t = read_time(path % i)
    x_grid = read_spatial_grid(path % i)
    data = read_probe_data(path % i)['scalars']['foo']
    
    for i, p in enumerate(itertools.product(*reversed(x_grid))):
        x = tuple(reversed(p))
        true = f(((t, x)))
        computed = data[i]
        error = abs(true - computed)
        max_error = max(error, max_error)
        if error > 1E-10:
            failed = True
            print '\t', i, '@', x, 'is', true, computed
assert not failed
print 'Max error', max_error

# With DG0 interpolation the error shold be rougly the mesh size 2^6
# so about 0.1
path = './results/sine_%d_0.vtk'

failed = False
max_error = 0
for i in range(15):
    t = read_time(path % i)
    x_grid = read_spatial_grid(path % i)
    data = read_probe_data(path % i)['scalars']['foo']
    
    for i, p in enumerate(itertools.product(*reversed(x_grid))):
        x = tuple(reversed(p))
        true = f(((t, x)))
        computed = data[i]
        error = abs(true - computed)
        max_error = max(max_error, error)

        if error > 2*1E-1:
            failed = True
            print '\t', i, '@', x, 'is', true, computed
assert not failed

print 'Max error', max_error
