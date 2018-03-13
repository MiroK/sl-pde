from dolfin import Timer, info
import numpy as np
import itertools


class StructuredGrid(object):
    '''
    A box [ll, ur] is partitioned into prod(ns) points to form a structured 
    grid
    '''
    def __init__(self, ll, ur, ns):
        assert len(ll) == len(ur) == len(ns)
        
        self.ll = np.fromiter(ll, dtype=float)
        self.ur = np.fromiter(ur, dtype=float)
        self.ns = np.fromiter(ns, dtype=int)

    def points(self):
        '''Probing points; first axis cycles fastest - like in VTK'''
        iters = [np.linspace(self.ll[i], self.ur[i], self.ns[i])
                 for i in range(len(self.ns))]

        for p in itertools.product(*reversed(iters)):
            yield np.fromiter(reversed(p), dtype=float)


class Probe(object):
    '''
    Setup probe for fields. This assumes that the mesh of the fields 
    stays constant and this things can be precomputed.
    '''
    def __init__(self, grid, fields):
        timer = Timer('setup'); timer.start()
        mesh = fields[0].function_space().mesh()
        # Same meshes for all
        assert all(mesh.id() == f.function_space().mesh().id() for f in fields)

        # Locate each point
        limit = mesh.num_entities_global(mesh.topology().dim())
        bbox_tree = mesh.bounding_box_tree()

        npoints = np.prod(grid.ns)
        cells_for_x = [None]*npoints
        for i, x in enumerate(grid.points()):
            cell = bbox_tree.compute_first_entity_collision(Point(*x))
            if -1 < cell < limit:
                cells_for_x[i] = Cell(mesh, cell)
        assert not any(c is None for c in cells_for_x)

        # For each field I want to build a function which which evals
        # it at all the points
        self.data = {}
        self.eval_fields = []
        for u in fields:
            # Alloc
            key = u.name()
            self.data[key] = [np.zeros(npoints) for _ in range(u.value_size())]
            # Attach the eval for u
            self.eval_fields.append(eval_u(u, grid.points(), cells_for_x, self.data[key]))
        info('Probe setup took %g' % timer.stop())
        self.grid = grid
        # Get values at construction
        self.update()

    def update(self):
        '''Evaluate now (with the fields as they are at the moment)'''
        timer = Timer('update'); timer.start()
        status = [f() for f in self.eval_fields]
        info('Probe update took %g' % timer.stop())
        return all(status)
    
    def write_vtk(self, path):
        '''Write the evaluate data to VTK'''
        with open(path, 'w') as f:
            f.write('# vtk DataFile Version 2.0\n')
            f.write('Basilisk\n')
            f.write('ASCII\n')
            f.write('DATASET STRUCTURED_GRID\n')
            f.write('DIMENSIONS %d %d %d\n' % tuple(self.grid.ns))
            f.write('POINTS %d double\n' % np.prod(self.grid.ns))
            # Write the grid
            for x, y, z in self.grid.points():
                f.write('%.16f %.16f %.16f\n' % (x, y, z))

            if not self.data: return None

            f.write('POINT_DATA %d\n' % np.prod(self.grid.ns))
            # Let's write the scalars
            for name, values in self.data.iteritems():
                if len(values) == 1:
                    f.write('SCALARS %s double\n' % name)
                    f.write('LOOKUP_TABLE default\n')

                    for val in values[0]: f.write('%.16f\n' % val)

            # Now vectors
            for name, values in self.data.iteritems():
                if len(values) == 3:
                    f.write('VECTORS %s double\n' % name)

                    for vals in zip(values[0], values[1], values[2]):
                        f.write('%.16f %.16f %.16f\n' % vals)

  
def eval_u(u, points, cells, data):
    '''Build a function for evaluating u in points'''    
    value_size = u.value_size()
    assert value_size == len(data)
    assert all(len(datai) == len(cells) for datai in data)

    V = u.function_space()
    element = V.dolfin_element()
    dim = element.space_dimension()
    # Precompute the basis matrix at x, and restrictions for the cell
    # that has x
    restrictions, basis_mats = [], []
    
    coefficients = np.zeros(dim)
    for x, ci in zip(points, cells):
        basis_matrix = np.zeros(value_size*dim)

        vc, orientation = ci.get_vertex_coordinates(), ci.orientation()
        # Eval the basis once
        element.evaluate_basis_all(basis_matrix, x, vc, orientation)

        basis_mats.append(basis_matrix.reshape((dim, value_size)).T)
        # Get the dofs
        restrictions.append(
            lambda cell=ci, coords=vc: u.restrict(coefficients, element, cell, coords, cell)
        )

    # The function
    def foo(mats=basis_mats, calls=restrictions, c=coefficients, data=data):
        for col, (A, f) in enumerate(zip(mats, calls)):
            f()  # Update
            for row, value in enumerate(A.dot(c)):
                data[row][col] = value
        return True
    return foo

# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import *
    from read_vtk import read_vtk

    vector = Expression(('A*x[0]', '-A*x[1]', '2*A*x[2]'), degree=1, A=1)
    scalar = Expression('A*(x[0]+2*x[1]+3*x[2])', degree=1, A=1)
    
    mesh = UnitCubeMesh(10, 10, 10)
    V = VectorFunctionSpace(mesh, 'CG', 1)
    S = FunctionSpace(mesh, 'CG', 1)

    v = interpolate(vector, V)
    v.rename('v', '0')
    
    f = interpolate(scalar, S)
    f.rename('f', '0')
    
    grid = StructuredGrid([0.2, ]*3, [0.8, ]*3, [25, 21, 18])

    probe = Probe(grid, (v, f))
    # Update to new expression
    vector.A = 2
    scalar.A = -1
    # # And functions
    v.assign(interpolate(vector, V))
    f.assign(interpolate(scalar, S))

    probe.update()
    
    # Write
    probe.write_vtk('test.vtk')
    # Read back
    data = read_vtk('test.vtk')

    # Check that this is fine
    grid = zip(*(xi.flatten() for xi in data['grid']))

    foo = (data['f'][0]).flatten()
    error = [abs(value - f(*x)) for x, value in zip(grid, foo)]
    assert max(error) < 5E-15, ([(value, f(*x)) for x, value in zip(grid, foo)], max(error))

    foo = zip(*(xi.flatten() for xi in data['v']))
    error = [np.linalg.norm(value - v(*x)) for x, value in zip(grid, foo)]
    assert max(error) < 5E-15, ([(value, v(*x)) for x, value in zip(grid, foo)], max(error))
