# cmake_minimum_required(VERSION <specify CMake version here>)
project(Intro_to_HPC)

set(CMAKE_CXX_STANDARD 14)

add_executable(HPC
        input.dat
        SEQ_Poisson.c
        MPI_Poisson.c)
