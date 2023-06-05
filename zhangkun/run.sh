#!/bin/bash

make clean
make

mtx=thermal2

a=${1:-1}
p=4
echo $a

if [ $a == 1 ]
then
    ./cg ${mtx}
elif [ $a == 2 ]
then
    mpirun -n ${p} -machinefile ./gpus ./mpi_cg ${mtx}
elif [ $a == 3 ]
then 
    mpirun -n ${p} -machinefile ./gpus ./mpi_cuda_cg ${mtx}
elif [ $a == 4 ]
then
    ./p1cg ${mtx}
elif [ $a == 5 ]
then
    ./p2cg ${mtx}
elif [ $a == 6 ]
then
    mpirun -n ${p} -machinefile ./gpus ./mpi_cg1 ${mtx}
elif [ $a == 7 ]
then
    mpirun -n ${p} -machinefile ./gpus ./mpi_cuda_cg1 ${mtx}
fi


