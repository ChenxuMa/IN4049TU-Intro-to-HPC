mpicc -o MPI_Poisson MPI_Poisson.c
prun -v -np 2 -4 -sge-script $PRUN_ETc/prun-openmpi ./MPI_Poisson 2 4
