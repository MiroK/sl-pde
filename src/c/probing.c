#include "probing.h"

vector field[];
vector pos[];
scalar foo[];
scalar bar[];

int main(){
  char path[256];

  GPoint ll = {0.2, 0.2, 0.2};
  GPoint ur = {0.8, 0.8, 0.8};
  Probe probe = {ll, ur, 4, 3, 5};

  // Quite large!
  init_grid(1 << 6);

  foreach(){
    field.x[] = x;
    field.y[] = -y;
    field.z[] = 2*z;

    pos.x[] = x;
    pos.y[] = y;
    pos.z[] = z;

    foo[] = x + y + z;
    bar[] = x + 2*y + 3*z;
  }

  double time = 123.45;

  // NOTE: vtk ending is added inside
  // first flat if 1/0 for binary the other is 0/1 for mpi-reduction
  // first flat if 1/0 for binary the other is 0/1 for mpi-reduction
  
  // NOTE: we store the data as structured points. When we query the function
  // the for interpolation the value we get is the one at the query point.
  // But othewise it's the value at center node of the cell. So WITOUT
  // interpolation we have error in the position
  sprintf(path, "sine_bin_0");
  probe_fields(probe, {foo, bar}, {field, pos}, path, time, 1, 0);

  sprintf(path, "sine_0");
  probe_fields(probe, {foo, bar}, {field, pos}, path, time, 0, 0);

  // Exact
  sprintf(path, "isine_bin_0");
  iprobe_fields(probe, {foo, bar}, {field, pos}, path, time, 1, 0);

  sprintf(path, "isine_0");
  iprobe_fields(probe, {foo, bar}, {field, pos}, path, time, 0, 0);
}
