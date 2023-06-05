
OBJ=mpi_cuda_cg.o cuda_cg.o
OBJ1=mpi_cuda_cg1.o cuda_cg1.o
LDFLAGS=-L/usr/local/cuda/lib64
LIB=-lcudart -lcurand

ALL:cg mpi_cg mpi_cuda_cg mpi_cg1 p1cg p2cg mpi_cuda_cg1

cg:cg.cpp
	g++ $^ -o $@

p1cg:p1cg.cpp
	g++ $^ -o $@

p2cg:p2cg.cpp
	g++ $^ -o $@

mpi_cg:mpi_cg.cpp
	mpic++ $^ -o $@

mpi_cg1:mpi_cg1.cpp
	mpic++ $^ -o $@

mpi_cuda_cg.o:mpi_cuda_cg.cpp
	mpic++ -c $^

cuda_cg.o:cuda_cg.cu
	nvcc -c $^

mpi_cuda_cg:$(OBJ)
	mpic++ $^ $(LDFLAGS) $(LIB) -o $@


mpi_cuda_cg1.o:mpi_cuda_cg1.cpp
	mpic++ -c $^

cuda_cg1.o:cuda_cg1.cu
	nvcc -c $^

mpi_cuda_cg1:$(OBJ1)
	mpic++ $^ $(LDFLAGS) $(LIB) -o $@


clean:
	rm -f cg mpi_cg mpi_cuda_cg $(OBJ) $(OBJ1)  mpi_cg1 p1cg p2cg mpi_cuda_cg1
