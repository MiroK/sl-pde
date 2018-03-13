import os, re
import numpy as np


def read_vtk(path):
    '''Read a VTK file produced by Basilisk'''
    # Fail early
    assert os.path.exists(path)

    nam, ext = os.path.splitext(path)
    assert ext == '.vtk'

    number_pattern = re.compile(r'[0-9]+')
    word_pattern = re.compile(r'\w+')

    data = {}
    f = open(path, 'r')
    lines = iter(f)
    while True:
        # Done?
        try:
            line = next(lines).strip('\n')
        except StopIteration:
            break

        # Dimensions line -> get the size
        if line.startswith('DIMENSIONS'):
            nx, ny, nz = map(int, number_pattern.findall(line))
            continue

        # Point line signals reading the grid
        if line.startswith('POINTS'):
            npoints = map(int, number_pattern.findall(line)).pop()
            assert npoints == (nx*ny*nz)
            
            data['grid'] = read_data(lines, nx, ny, nz, 3)
            continue

        # For scalar the second guy in the line is the field name and
        # The next line is a colormap which can be skipped
        if line.startswith('SCALARS'):
            _, field_name, _ = word_pattern.findall(line)
            next(lines)  # Skip
            
            data[field_name] = read_data(lines, nx, ny, nz, 1)
            continue

        # Finally vector
        if line.startswith('VECTORS'):
            # VECTORS field.x type -> VECTORS, field, x type
            _, field_name, _, _ = word_pattern.findall(line)
            
            data[field_name] = read_data(lines, nx, ny, nz, 3)
            continue
    f.close()

    return data


def read_data(lines, nx, ny, nz, N):
    '''Consume nx*ny*nz lines from the iterator. Each line is N-tuple'''
    data = [np.zeros((nx, ny, nz)) for i in range(N)]

    number_pattern = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                line = next(lines)

                values = map(float, number_pattern.findall(line))
                for component, value in enumerate(values):
                    data[component][i, j, k] = value
    return data

# --------------------------------------------------------------------

if __name__ == '__main__':
    # NOTE: we expect that VTK data has vector field which is (x, -y, 2z)
    # scalars foo = x+y+z, bar=x+2y+3z

    # Begin with isine which should is using linear interpolation. Since
    # we're using linear functions in the tests we should be exact
    data = read_vtk('../c/isine_0.vtk')

    grid = zip(*(xi.flatten() for xi in data['grid']))

    foo = (data['foo'][0]).flatten()
    assert max(abs(value - (x+y+z)) for ((x, y, z), value) in zip(grid, foo)) < 1E-15
    
    bar = (data['bar'][0]).flatten()
    assert max(abs(value - (x+2*y+3*z)) for ((x, y, z), value) in zip(grid, bar)) < 1E-15

    field = zip(*(xi.flatten() for xi in data['field']))
    assert max(np.linalg.norm(np.array([x, -y, 2*z]) - np.array(value))
               for ((x, y, z), value) in zip(grid, field)) < 1E-15

    # When we do piecewise constant interpolation there are bound to be
    # some differences but should be small
    data = read_vtk('../c/sine_0.vtk')

    grid = zip(*(xi.flatten() for xi in data['grid']))

    foo = (data['foo'][0]).flatten()
    assert max(abs(value - (x+y+z)) for ((x, y, z), value) in zip(grid, foo)) < 2E-2
    
    bar = (data['bar'][0]).flatten()
    assert max(abs(value - (x+2*y+3*z)) for ((x, y, z), value) in zip(grid, bar)) < 2E-2

    field = zip(*(xi.flatten() for xi in data['field']))
    assert max(np.linalg.norm(np.array([x, -y, 2*z]) - np.array(value))
               for ((x, y, z), value) in zip(grid, field)) < 2E-2
