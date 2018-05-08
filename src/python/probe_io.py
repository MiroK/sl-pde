from vtk import vtkStructuredPointsReader as VTKReader
from vtk import vtkStructuredPointsWriter as VTKWriter
from vtk import vtkStructuredPoints, vtkPointData
from vtk.util import numpy_support as vtk_np
import numpy as np
import re, os


number = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')


def read_time(path):
    '''Extract timestamp from file'''
    reader = VTKReader()
    reader.SetFileName(path)
    reader.Update()
    
    header = reader.GetHeader()
    return float(number.findall(header)[0])


def read_spatial_grid(path):
    '''Construct axes of the spatial grid which is a tensor product'''
    reader = VTKReader()
    reader.SetFileName(path)
    reader.Update()

    grid = reader.GetOutput()
    
    os = grid.GetOrigin()
    dxs = grid.GetSpacing()
    sizes = grid.GetDimensions()

    return [(o + np.linspace(0, (size-1)*dx, size)).tolist() for (o, dx, size) in zip(os, dxs, sizes)]
        

def read_probe_data(path, scalars=(), vectors=()):
    '''Point data the specified vars in the probe'''
    reader = VTKReader()
    reader.SetFileName(path)
    reader.Update()
    # All if not specified
    if not scalars:
        scalars = []
        i = 0
        while reader.GetScalarsNameInFile(i):
            scalars.append(reader.GetScalarsNameInFile(i))
            i += 1
    # All if not speciied
    if not vectors:
        vectors = []
        i = 0
        while reader.GetVectorsNameInFile(i):
            vectors.append(reader.GetVectorsNameInFile(i))
            i += 1

    scalars_data = {}
    for scalar in scalars:
        reader.SetScalarsName(scalar)
        reader.Update()
        data = vtk_np.vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(scalar))
        scalars_data[scalar] = data

    vectors_data = {}
    for vector in vectors:
        reader.SetVectorsName(vector)
        reader.Update()

        data = vtk_np.vtk_to_numpy(reader.GetOutput().GetPointData().GetArray(vector))
        vectors_data[vector] = data

    data = {}
    if scalars_data: data['scalars'] = scalars_data

    if vectors_data: data['vectors'] = vectors_data

    return data


def minimum(*arrays):
    '''Reduce several array by min'''
    assert len(arrays) > 1
    # The real case
    if len(arrays) == 2:
        return np.minimum(*arrays)
    # Recurse
    return minimum(*((minimum(arrays[0], arrays[1]), ) + arrays[2:]))


def merge_mpi_vtk(files, result_dir, write_binary=True):
    '''
    VTK files could have been written on several CPUs having HUGE as a 
    value for points which didn't belong to the process. Merge does equivalent
    of MPI.Min for the results and saves the data for VTK.
    '''
    assert files
    assert len(files) > 1,  '%s is plenty merged, use copy?' % files[0]

    # Check probe sanity
    grid = read_spatial_grid(files[0])
    assert all(grid == read_spatial_grid(f) for f in files)

    # Time
    times = map(read_time, files)
    time, dt = np.mean(times), np.std(times)
    if dt > 1E-12: print 'Adjusted time'

    # Consistency and reduction of data
    datas = [read_probe_data(f, (), ()) for f in files]
    # vectors/scalars are in both
    keys, = set(tuple(d.keys()) for d in datas)
    # Merge vector/scalar:var picking var from first d (assume same)
    merged_data = {}
    for tensor in keys:
        # Fail on key missing, shape mismatch
        vec_data = {var: minimum(*[d[tensor][var] for d in datas])
                    for var in datas[0][tensor]}
        merged_data[tensor] = vec_data

    # Data
    header_data = {'origin': tuple(g[0] for g in grid),
                   'sizes': tuple(map(len, grid)),
                   'spacing': tuple((g[1] - g[0]) for g in grid),
                   'time': time}

    out = os.path.join(result_dir, os.path.basename(files[0]))

    write_structured_points(out, header_data, merged_data, write_binary)
    
    return 'Merged %r to %s' % (files, out)


def write_structured_points(out, header_data, data, write_binary=True):
    '''Save as VTK's structured points'''
    # FIXME: I can't get this work properly with vtk so we write manually
    with open(out, 'wb') as f:
        f.write('# vtk DataFile Version 2.0\n')
        f.write('Basilisk %.16f\n' % header_data['time'])
        f.write('BINARY\n' if write_binary else 'ASCII\n')
        f.write('DATASET STRUCTURED_POINTS\n')
        f.write('DIMENSIONS %d %d %d\n' % header_data['sizes'])
        f.write('ORIGIN %.16f %.16f %.16f\n' % header_data['origin'])
        f.write('SPACING %.16f %.16f %.16f\n' % header_data['spacing'])

        f.write('POINT_DATA %d\n' % np.prod(header_data['sizes']))
        
        if 'scalars' in data:
            for var in data['scalars']:
                values = data['scalars'][var]
                f.write('SCALARS %s double\n' % var)
                f.write('LOOKUP_TABLE defaults\n')

                if write_binary:
                    values.astype(values.dtype.newbyteorder('>')).tofile(f, sep='')  # Bigendian
                else:
                    for v in values: f.write('%.16f\n' % v)
        f.write('\n')
        
        if 'vectors' in data:
            for var in data['vectors']:
                values = data['vectors'][var]
                npoints, ncomps = values.shape
                assert 1 < ncomps < 4
                if ncomps == 2:
                    values = np.c_[values, np.zeros(npoints)]
                
                f.write('VECTORS %s double\n' % var)

                if write_binary:
                    values.astype(values.dtype.newbyteorder('>')).tofile(f, sep='')
                else:
                    for v in values: f.write('%.16f %.16f %.16f\n' % tuple(v))
        f.write('\n')

# -------------------------------------------------------------------

if __name__ == '__main__':
    from functools import partial
    import argparse, os

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # So sine_0_*
    parser.add_argument('root', type=str, help='Root file template')
    
    # Discovery sine_
    discover_parser = parser.add_mutually_exclusive_group(required=False)
    discover_parser.add_argument('--discover', dest='discover', action='store_true')
    discover_parser.add_argument('--no-discover', dest='discover', action='store_false')

    # Storing
    parser.add_argument('--dir', default='.', help='Where to store the merged results')
      
    parser.set_defaults(discover=False)

    args = parser.parse_args()
    

    if not os.path.exists(args.dir): os.mkdir(args.dir)

    # We will perform merge on files computed on different ranks
    source_dir = os.path.dirname(args.root)
    candidates = map(lambda f: os.path.join(source_dir, f), os.listdir(source_dir))
    # We have to discover time and rank; disover time
    if args.discover:
        pattern = re.compile(r'%s_\w+.vtk' % args.root)
        files = filter(lambda f: pattern.match(f) is not None, candidates)
        assert files
        # Isolate times
        mpi_templates = list(set(f[:f.rfind('_')] for f in files))
    # No need for time, template is ready
    else:
        mpi_templates = [args.root]
    # Patterns for mpi
    mpi_templates = [r'%s_\d+.vtk' % t for t in mpi_templates]

    mpi_templates = map(re.compile, mpi_templates)

    # Discover ranks for tamplate
    for template in mpi_templates:
        mpi_files = filter(lambda f, template=template: template.match(f) is not None, candidates)
        assert mpi_files
        
        # Sort by mpi rank so that taking first will give us name for the merger
        mpi_files = sorted(mpi_files, key=lambda f: float(number.findall(f)[-1]))

        print merge_mpi_vtk(mpi_files, args.dir)
