run.x: probing.c
	qcc -source -D_MPI=1 probing.c
	mpicc -Wall -O2 -std=c99 _probing.c -o run.x -lm

