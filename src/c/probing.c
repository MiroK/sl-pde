#include "grid/octree.h"
#include "vtk.h"
#include <math.h>


typedef struct {
  // This is a geometric point
  double x;
  double y;
  double z;
} GPoint;


typedef struct {
  // A grid of points defined by ll corner and ur corner and resolutions
  // in the directions
  GPoint ll;
  GPoint ur;
  int nx;
  int ny;
  int nz;
} Probe;


void write_probedata_vtk(double** data, Probe probe, scalar* scalar_fields, vector* vector_fields, char* path){
  int nrows = 0;
  for(scalar field in scalar_fields){
    nrows += 1;
  }

  for(vector field in vector_fields){
    nrows += 3;
  }

  int ncols = probe.nx*probe.ny*probe.nz;

  const double dx = (probe.ur.x - probe.ll.x)/(probe.nx-1);
  const double dy = (probe.ur.y - probe.ll.y)/(probe.ny-1);
  const double dz = (probe.ur.z - probe.ll.z)/(probe.nz-1);
  
  double x, y, z;
  
  FILE* fp;
  fp = fopen(path, "w");

  fputs ("# vtk DataFile Version 2.0\n"
         "Basilisk\n"
         "ASCII\n"
         "DATASET STRUCTURED_GRID\n", fp);
  fprintf(fp, "DIMENSIONS %d %d %d\n", probe.nx, probe.ny, probe.nz);
  fprintf(fp, "POINTS %d double\n", ncols);

  for(int k = 0; k < probe.nz; k++){
    z = probe.ll.z + k*dz;
    for(int j = 0; j < probe.ny; j++){
      y = probe.ll.y + j*dy;
      for(int i = 0; i < probe.nx; i++){
        x = probe.ll.x + i*dx;

        fprintf (fp, "%g %g %g\n", x, y, z);
      }
    }
  }

  fprintf (fp, "POINT_DATA %d\n", ncols);
  
  int row_index = 0;
  for(scalar field in scalar_fields){
    fprintf (fp, "SCALARS %s double\n", field.name);
    fputs ("LOOKUP_TABLE default\n", fp);
    for(int col=0; col<ncols; col++){
      fprintf(fp, "%g\n", data[row_index][col]);
    }
    row_index += 1;
  }

  for(vector field in vector_fields){
    fprintf (fp, "VECTORS %s double\n", field.x.name);
    for(int col=0; col<ncols; col++){
      fprintf(fp, "%g %g %g\n", data[row_index][col], data[row_index+1][col], data[row_index+2][col]);
    }
    row_index += 3;
  }
  fflush (fp);
  fclose(fp);
}


void probe_fields(Probe probe, scalar* scalar_fields, vector* vector_fields, char* path){
  /*
    Here the value of the field is computed by taking the cell value 
    at a cell which contains the probe points. This allows reusing the 
    info (about cell) for all the fields but is less accurate.
   */
  //How much data do I need
  //A row in data cooresponds to evals at all pts of a scalar component
  int nrows = 0;
  for(scalar field in scalar_fields){
    nrows += 1;
  }

  for(vector field in vector_fields){
    nrows += 3;
  }
  // Each column is a different point
  int ncols = probe.nx*probe.ny*probe.nz;
  
  double **data = malloc(nrows*sizeof(double *));
  for(int i=0; i<nrows; i++){
    data[i] = malloc(ncols*sizeof(double));
  }

  const double dx = (probe.ur.x - probe.ll.x)/(probe.nx-1);
  const double dy = (probe.ur.y - probe.ll.y)/(probe.ny-1);
  const double dz = (probe.ur.z - probe.ll.z)/(probe.nz-1);

  double x, y, z;
  // Let each process will the local grid
  int row_index = 0;
  int col_index = 0;
  for(int k = 0; k < probe.nz; k++){
    z = probe.ll.z + k*dz;
    for(int j = 0; j < probe.ny; j++){
      y = probe.ll.y + j*dy;
      for(int i = 0; i < probe.nx; i++){
        x = probe.ll.x + i*dx;

        // Locate once
        Point point = locate (x, y, z);
        // Use to fill scalar data
        for(scalar field in scalar_fields){
          data[row_index][col_index] = (point.level >= 0) ? val(field) : nodata;
          row_index += 1;
        }

        for(vector field in vector_fields){
          // Components
          data[row_index][col_index] = (point.level >= 0) ? val(field.x) : nodata;
          row_index += 1;

          data[row_index][col_index] = (point.level >= 0) ? val(field.y) : nodata;
          row_index += 1;

          data[row_index][col_index] = (point.level >= 0) ? val(field.z) : nodata;
          row_index += 1;
        }
        // Onto next point
        row_index = 0;  // Reset the row count for fields
        col_index += 1;
      }
    }
  }

  // We perform reduction on 0 node. Note that if the point was no found
  // we have nodata(HUGE) on cpu. MPI_MIN syncs things
  if (pid() == 0){
    for(int row=0; row<nrows; row++){
      MPI_Reduce(MPI_IN_PLACE, data[row], ncols, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
  }
  else{
    for(int row=0; row<nrows; row++){
      MPI_Reduce(data[row], data[row], ncols, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
  }

  // Write data to one VTK file on root - the one process that should have
  // the right values
  if(pid() == 0){
    write_probedata_vtk(data, probe, scalar_fields, vector_fields, path);
  }
  
  for(int i=0; i<nrows; i++){
    free(data[i]);
  }
  free(data);
}


void iprobe_fields(Probe probe, scalar* scalar_fields, vector* vector_fields, char* path){
  /*
    Here the value of the field is computed by interpolation. It should be 
    more accurate but also expensive.
   */
  //How much data do I need
  //A row in data cooresponds to evals at all pts of a scalar component
  int nrows = 0;
  for(scalar field in scalar_fields){
    nrows += 1;
  }

  for(vector field in vector_fields){
    nrows += 3;
  }
  // Each column is a different point
  int ncols = probe.nx*probe.ny*probe.nz;
  
  double **data = malloc(nrows*sizeof(double *));
  for(int i=0; i<nrows; i++){
    data[i] = malloc(ncols*sizeof(double));
  }

  const double dx = (probe.ur.x - probe.ll.x)/(probe.nx-1);
  const double dy = (probe.ur.y - probe.ll.y)/(probe.ny-1);
  const double dz = (probe.ur.z - probe.ll.z)/(probe.nz-1);

  double x, y, z, value;
  // Let each process will the local grid
  int row_index = 0;
  int col_index = 0;

  // Use to fill scalar data
  for(scalar field in scalar_fields){
    // Every point
    for(int k = 0; k < probe.nz; k++){
      z = probe.ll.z + k*dz;
      for(int j = 0; j < probe.ny; j++){
        y = probe.ll.y + j*dy;
        for(int i = 0; i < probe.nx; i++){
          x = probe.ll.x + i*dx;

          value = interpolate(field, x, y, z);
          data[row_index][col_index] = (value < nodata) ? value : nodata;
          // Next point
          col_index += 1;
        }
      }
    }
    // Next field
    col_index = 0;
    row_index += 1;
  }

  for(vector field in vector_fields){
    // Every point
    for(int k = 0; k < probe.nz; k++){
      z = probe.ll.z + k*dz;
      for(int j = 0; j < probe.ny; j++){
        y = probe.ll.y + j*dy;
        for(int i = 0; i < probe.nx; i++){
          x = probe.ll.x + i*dx;

          value = interpolate(field.x, x, y, z);
          data[row_index][col_index] = (value < nodata) ? value : nodata;

          value = interpolate(field.y, x, y, z);
          data[row_index+1][col_index] = (value < nodata) ? value : nodata;

          value = interpolate(field.z, x, y, z);
          data[row_index+2][col_index] = (value < nodata) ? value : nodata;
          // Next point
          col_index += 1;
        }
      }
    }
    // Next field
    col_index = 0;
    row_index += 3;
  }

  // We perform reduction on 0 node. Note that if the point was no found
  // we have nodata(HUGE) on cpu. MPI_MIN syncs things
  if (pid() == 0){
    for(int row=0; row<nrows; row++){
      MPI_Reduce(MPI_IN_PLACE, data[row], ncols, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
  }
  else{
    for(int row=0; row<nrows; row++){
      MPI_Reduce(data[row], data[row], ncols, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
  }

  // Write data to one VTK file on root - the one process that should have
  // the right values
  if(pid() == 0){
    write_probedata_vtk(data, probe, scalar_fields, vector_fields, path);
  }
  
  for(int i=0; i<nrows; i++){
    free(data[i]);
  }
  free(data);
}



//////////////////////////////////////////////////////////////////////
// TEST
//////////////////////////////////////////////////////////////////////


#define funcint(x, y, z) (sin(M_PI*((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5))))

vector field[];
scalar foo[];
scalar bar[];

int main(){
  char path[256];

  GPoint ll = {0.2, 0.2, 0.2};
  GPoint ur = {0.8, 0.8, 0.8};
  Probe probe = {ll, ur, 4, 3, 5};
  
  init_grid(1 << 5);

  double value;
  foreach(){
    value = funcint(x, y, z);
    field.x[] = value;
    field.y[] = 2 + value;
    field.z[] = 2*value;

    foo[] = 1 - value;
    bar[] = 10 * value;
  }

  sprintf(path, "sine_0.vtk");  
  probe_fields(probe, {foo, bar}, {field}, path);

  sprintf(path, "isine_0.vtk");  
  iprobe_fields(probe, {foo, bar}, {field}, path);
}
