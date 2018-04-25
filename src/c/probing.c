#include "probing.h"

#define funcint(x, y, z) (sin(M_PI*((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5))))

vector field[];
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

    foo[] = x + y + z;
    bar[] = x + 2*y + 3*z;
  }

  double time = 123.45;

  // Reduce in parallel and write on precess
  // NOTE: vtk ending is added inside
  sprintf(path, "sine_0");
  // first flat if 1/0 for binary the other is 0/1 for mpi-reduction
  probe_fields(probe, {foo}, {field}, path, time, 1, 0);
  
  //sprintf(path, "isine_0");  
  //iprobe_fields(probe, {foo, bar}, {field}, path, time);
}
