run.x: probing.c probing.h
	qcc -source -D_MPI=1 -DTRACE=2 probing.c
	mpicc -Wall -O2 -std=c99 _probing.c -o run.x -lm

