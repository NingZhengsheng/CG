#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>

/*host variable*/
extern int *ptr;
extern int *col;
extern double *val;
extern double *b;
extern double *x;

// iterative variable
extern double *r0;
extern double *r1;
extern double *z0;
extern double *z1;
extern double beta;
extern double alpha;
extern double *p;
extern double *Ap;
// subvector
extern double *r1_sub;
extern double *p_sub;
// dot temp
extern double *s;

extern double sub_norm;

extern int num_elements, num_rows;
// mpi variable
extern int worldRank, worldSize, rowPart, lastPart;
/******************************************************/
/*device variable*/
extern int *d_ptr;
extern int *d_col;
extern double *d_val;
extern double *d_b;
extern double *d_x;
// iterative variable
extern double *d_r0;
extern double *d_r1;
extern double *d_z0;
extern double *d_z1;
extern double *d_p;
extern double *d_Ap;
extern double *d_r1_sub;
extern double *d_p_sub;
// dot temp
extern double *d_s;


/*cudaMalloc*/
void initGPU() {
  cudaError_t err;
  /*malloc*/
  cudaMalloc((void **)&d_ptr, sizeof(int) * (num_rows + 1));
  cudaMalloc((void **)&d_col, sizeof(int) * (num_elements));
  cudaMalloc((void **)&d_val, sizeof(double) * (num_elements));
  cudaMalloc((void **)&d_b, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_x, sizeof(double) * (num_rows));

  cudaMalloc((void **)&d_r0, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_r1, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_z0, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_z1, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_p, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_Ap, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_r1_sub, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_p_sub, sizeof(double) * (num_rows));

  /*cudaHostAlloc*/
  cudaHostAlloc((void**)&p_sub,sizeof(double) * (num_rows),cudaHostAllocDefault);
  cudaHostAlloc((void**)&r1_sub,sizeof(double) * (num_rows),cudaHostAllocDefault);
  cudaHostAlloc((void**)&p,sizeof(double) * (num_rows),cudaHostAllocDefault);
  cudaHostAlloc((void**)&x,sizeof(double) * (num_rows),cudaHostAllocDefault);
  cudaHostAlloc((void**)&r1,sizeof(double) * (num_rows),cudaHostAllocDefault);
  cudaHostAlloc((void**)&z0,sizeof(double) * (num_rows),cudaHostAllocDefault);
  cudaHostAlloc((void**)&z1,sizeof(double) * (num_rows),cudaHostAllocDefault);
  for (int i = 0; i < num_rows; i++)//initialize x and b
	{
		x[i] = 1;
		b[i] = 1;
	}
  /*memcpy*/
  cudaMemcpy(d_ptr, ptr, sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col, col, sizeof(int) * (num_elements), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, val, sizeof(double) * (num_elements),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);
  err = cudaMemcpy(d_b, b, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("CUDA error: %s \n", cudaGetErrorString(err));
  }
} 

/*data   r0=b - Ax  p0 = r0*/
__global__ void data(int *d_ptr, int *d_col, double *d_val, double *d_b,
                     double *d_x, double *d_r0, double *d_r1_sub, double *d_p_sub,
                     int num_rows, int offset) {
  int row = blockDim.x * blockIdx.x + threadIdx.x + offset;
  if (row < num_rows) {
    double sum_of_row = 0;
    int row_start = d_ptr[row];
    int row_end = d_ptr[row + 1];

    for (int i = row_start; i < row_end; i++) {
      sum_of_row += d_val[i] * d_x[d_col[i]];
    }
    d_r0[row] = d_b[row] - sum_of_row;
    d_r1_sub[row] = d_r0[row];
    //d_p_sub[row] = d_r0[row];
  }
}
void trans(double *a, double *b) //b=a
{
  cudaMemcpy(b, a, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);
}

/*init data   r0=b - Ax  p0 = r0*/
void initData() {
  int offset = worldRank * rowPart;
  int stride = rowPart;
  if(worldRank == (worldSize - 1))stride=rowPart + lastPart;
  int blockDim = 512;

  int gridDim = (stride % blockDim) > 0 ? (stride / blockDim) + 1
                                         : (stride / blockDim);

  // dot temp
  //s = new double[gridDim];
  cudaHostAlloc((void**)&s,sizeof(double) * (gridDim),cudaHostAllocDefault);
  cudaMalloc((void **)&d_s, sizeof(double) * (gridDim));
  data<<<gridDim, blockDim>>>(d_ptr, d_col, d_val, d_b, d_x, d_r0, d_r1_sub,
                              d_p_sub, num_rows, offset);
  if (lastPart != 0 && worldRank == (worldSize - 1)) {
    offset = worldSize * rowPart;
    gridDim = (lastPart % blockDim) > 0 ? (lastPart / blockDim) + 1
                                        : (lastPart / blockDim);
    data<<<gridDim, blockDim>>>(d_ptr, d_col, d_val, d_b, d_x, d_r0, d_r1_sub,
                                d_p_sub, num_rows, offset);
  }
  cudaMemcpy(r1_sub, d_r1_sub, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);
}

/*norm device*/
__global__ void residualD(int num_rows, double *r, double *s, int offset,
                          int stride) {
  __shared__ double cache1[512]; // r block 512
  int row = blockDim.x * blockIdx.x + threadIdx.x + offset;
  int cacheIndex = threadIdx.x;
  if (row < offset + stride) {
    cache1[cacheIndex] = r[row] * r[row];
  } else {
    cache1[cacheIndex] = 0;
  }
  __syncthreads();
  int i = blockDim.x / 2;
  //在块内计算部分和
  while (i != 0) {
    if (cacheIndex < i) {
      cache1[cacheIndex] += cache1[cacheIndex + i];
    }
    __syncthreads();
    i /= 2;
  }
  if (cacheIndex == 0) {
    s[blockIdx.x] = cache1[0];
  }
}
/*init norm*/
void initResidual() {
  int offset = worldRank * rowPart;
  int stride = rowPart;
  if(worldRank == (worldSize - 1))stride=rowPart + lastPart;

  int blockDim = 512;
  int gridDim = (stride % blockDim) > 0 ? (stride / blockDim) + 1
                                         : (stride / blockDim);
  residualD<<<gridDim, blockDim>>>(num_rows, d_r1_sub, d_s, offset, stride);
  cudaMemcpy(s, d_s, sizeof(double) * gridDim, cudaMemcpyDeviceToHost);
  sub_norm = 0;
  for (int i = 0; i < gridDim; i++) {
    sub_norm += s[i];
  }
}

/*Ap*/
__global__ void spmvAp(int *d_ptr, int *d_col, double *d_val, double *d_p,
                       double *d_Ap, int num_rows, int offset) {
  int row = blockDim.x * blockIdx.x + threadIdx.x + offset;
  if (row < num_rows) {
    double sum_of_row = 0;
    int row_start = d_ptr[row];
    int row_end = d_ptr[row + 1];

    for (int i = row_start; i < row_end; i++) {
      sum_of_row += d_val[i] * d_p[d_col[i]];
    }
    d_Ap[row] = sum_of_row;
  }
}
/*init Ap*/
void initAp() {

  cudaMemcpy(d_p, p, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);

  int offset = worldRank * rowPart;
  int blockDim = 512;
  int gridDim = (rowPart % blockDim) > 0 ? (rowPart / blockDim) + 1
                                         : (rowPart / blockDim);
  spmvAp<<<gridDim, blockDim>>>(d_ptr, d_col, d_val, d_p, d_Ap, num_rows,
                                offset);
  if (lastPart != 0 && worldRank == (worldSize - 1)) {
    offset = worldSize * rowPart;
    gridDim = (lastPart % blockDim) > 0 ? (lastPart / blockDim) + 1
                                        : (lastPart / blockDim);
    spmvAp<<<gridDim, blockDim>>>(d_ptr, d_col, d_val, d_p, d_Ap, num_rows,
                                  offset);
  }
}

/*SMV*/
__global__ void SMV(double *a, double *b, double c, double *d, int num_rows,
                    int offset, int rowPart, int lastPart) {
  int row = blockDim.x * blockIdx.x + threadIdx.x + offset;
  if (row < offset + rowPart + lastPart && row < num_rows) {
    a[row] = b[row] + c * d[row];
  }
}
/*init SMV*/
void initSMV(double *aa, double *a, double *b, double c,
             double *d) {
  int offset = worldRank * rowPart;
  int blockDim = 512;
  int gridDim = (rowPart % blockDim) > 0 ? (rowPart / blockDim) + 1
                                         : (rowPart / blockDim);
  SMV<<<gridDim, blockDim>>>(a, b, c, d, num_rows, offset, rowPart, lastPart);

  cudaMemcpy(aa, a, sizeof(double) * (num_rows), cudaMemcpyDeviceToHost);
}

/*replaceVector r0=r1*/
__global__ void replaceV(double *d_r0, double *d_r1, int num_rows, int offset,
                         int rowPart, int lastPart) {
  int row = blockDim.x * blockIdx.x + threadIdx.x + offset;
  if (row < offset + rowPart + lastPart && row < num_rows) {
    d_r0[row] = d_r1[row];
  }
}
/*init replaceVector   r0=r1*/
void initReplaceV(double *a,double *b) {
  int offset = worldRank * rowPart;
  int blockDim = 512;
  int gridDim = (rowPart % blockDim) > 0 ? (rowPart / blockDim) + 1
                                         : (rowPart / blockDim);
  replaceV<<<gridDim, blockDim>>>(a, b, num_rows, offset, rowPart,
                                  lastPart);
}

/*dot device*/
__global__ void multiV(int num_rows, double *a, double *b, double *s,
                       int offset, int stride) {
  __shared__ double cache1[512]; // r block 512
  int row = blockDim.x * blockIdx.x + threadIdx.x + offset;
  int cacheIndex = threadIdx.x;
  if (row < offset + stride) {
    cache1[cacheIndex] = a[row] * b[row];
  } else {
    cache1[cacheIndex] = 0;
  }
  __syncthreads();
  int i = blockDim.x / 2;
  //在块内计算部分和
  while (i != 0) {
    if (cacheIndex < i) {
      cache1[cacheIndex] += cache1[cacheIndex + i];
    }
    __syncthreads();
    i /= 2;
  }
  if (cacheIndex == 0) {
    s[blockIdx.x] = cache1[0];
  }
}

/*init dot device*/
void initMultiV(double *a, double *b) {
  //cudaMemcpy(b, bb, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);
  int offset = worldRank * rowPart;
  int stride = rowPart;
  if(worldRank == (worldSize - 1))stride=rowPart + lastPart;
  int blockDim = 512;
  int gridDim = (stride % blockDim) > 0 ? (stride / blockDim) + 1
                                         : (stride / blockDim);
  multiV<<<gridDim, blockDim>>>(num_rows, a, b,d_s, offset, stride);
  cudaMemcpy(s, d_s, sizeof(double) * gridDim, cudaMemcpyDeviceToHost);
  sub_norm = 0;
  for (int i = 0; i < gridDim; i++) {
    sub_norm += s[i];
  }
}