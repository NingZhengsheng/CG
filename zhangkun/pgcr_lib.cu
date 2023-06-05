#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cusparse.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

FILE *f1, *f2, *f3;                     // read data-> A,b,x0
int nx = 360, ny = 180, nz = 38;        // 3D scale
static const int size = 360 * 180 * 38; // size->360*180*38;
static const int NEIGHBOR = 19;         // neighbor->19
double *A, *b, *x;                      // variable
int num_elements = 0;                   // number of elements
int num_rows = 360 * 180 * 38;          // number of rows

int *ptr;
int *col;
double *val;
/*csr ilu0*/
double *val_m;
int *mid_index; //对角线元素index

double *Ap;
double *Ar;
double *r;
double *p;
double alpha = 0;
double *beta;
int iteration = 1;
double threshold = 1e-10; //��ֵ

//回代过程中间变量
double *y;
double *r1;

#define THREADS_PER_BLOCK 512
#define IDX(i, j, k) ((i)*ny * nz + (j)*nz + (k)) // idx of vector
#define IDXA(i, j, k, q)                                                       \
  ((i)*ny * nz * NEIGHBOR + (j)*nz * NEIGHBOR + (k)*NEIGHBOR +                 \
   (q)) // idx of matrix
#define MAX_ITERATION 150
int num_k = 10;
double serialLU = 0;
int main() {

  __host__ void readMatrix();
  __host__ void initData(cusparseHandle_t handle, cublasHandle_t handle2,
                         cusparseMatDescr_t descr, double *d_val, int *d_ptr,
                         int *d_col, double *d_x, double *d_b, double *d_r,
                         double *d_p, double *d_r1);
  __host__ void spmvAp(cusparseHandle_t handle, cusparseMatDescr_t descr,
                       double *d_val, int *d_ptr, int *d_col, double *d_p,
                       double *d_Ap);

  __host__ void calAlpha(cublasHandle_t handle2, double *d_r, double *d_Ap);
  __host__ void calXAndR(cublasHandle_t handle2, double *d_p, double *d_Ap,
                         double *d_r, double *d_x);

  __host__ void spmvAr(cusparseHandle_t handle, cusparseMatDescr_t descr,
                       double *d_val, int *d_ptr, int *d_col, double *d_r1,
                       double *d_Ar);

  __host__ void calBeta(cublasHandle_t handle2, double *d_Ar, double *d_Ap,
                        double *d_beta);

  __host__ void calPAndAp(cublasHandle_t handle2, double *d_p, double *d_Ap,
                          double *d_r1, double *d_Ar, double *p_temp,
                          double *Ap_temp);
  __host__ void memFree();

  /*ilu0*/
  __host__ void ilu0(); // ilu0分解
  __host__ void LU();   //三角回代
  // time start
  clock_t start_s, stop_s;
  double duration_s;

  start_s = clock();
  readMatrix();
  stop_s = clock();
  duration_s = (double)(stop_s - start_s) * 1000 / CLOCKS_PER_SEC;
  printf("read time: %.0lf (ms)\n", duration_s);
  ilu0(); // ilu分解

  int *d_ptr;
  int *d_col;
  double *d_val;

  double *d_b;
  double *d_x;
  double *d_Ap;
  double *d_Ar;
  double *d_r;
  double *d_r1;
  double *d_p;
  double *d_beta;
  double *p_temp;
  double *Ap_temp;

  cudaMalloc((void **)&d_ptr, sizeof(int) * (num_rows + 1));
  cudaMalloc((void **)&d_col, sizeof(int) * (num_elements));
  cudaMalloc((void **)&d_val, sizeof(double) * (num_elements));
  cudaMalloc((void **)&d_b, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_x, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_Ap, sizeof(double) * (num_rows * num_k));
  cudaMalloc((void **)&d_Ar, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_r, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_r1, sizeof(double) * (num_rows));
  cudaMalloc((void **)&d_p, sizeof(double) * (num_rows * num_k));
  cudaMalloc((void **)&d_beta, sizeof(double) * (num_k));
  cudaMalloc((void **)&p_temp, sizeof(double) * (num_rows));
  cudaMalloc((void **)&Ap_temp, sizeof(double) * (num_rows));

  cudaMemcpy(d_ptr, ptr, sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col, col, sizeof(int) * (num_elements), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, val, sizeof(double) * (num_elements),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ap, Ap, sizeof(double) * (num_rows * num_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, p, sizeof(double) * (num_rows * num_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, beta, sizeof(double) * (num_k), cudaMemcpyHostToDevice);
  cusparseHandle_t handle;
  cusparseMatDescr_t descr;
  cublasHandle_t handle2;

  /* initialize cusparse and cublas library */
  cusparseStatus_t stat1 = cusparseCreate(&handle);
  if (stat1 != CUSPARSE_STATUS_SUCCESS) {
    printf("CUSPARSE initialization failed\n");
    return EXIT_FAILURE;
  }
  cublasStatus_t stat2 = cublasCreate(&handle2);
  if (stat2 != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }
  /* create and setup matrix descriptor */
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  int blockNum = (num_rows % THREADS_PER_BLOCK) > 0
                     ? (num_rows / THREADS_PER_BLOCK) + 1
                     : (num_rows / THREADS_PER_BLOCK);

  cout << "blockNum:" << blockNum << endl;
  cout << "***********" << endl;

  initData(handle, handle2, descr, d_val, d_ptr, d_col, d_x, d_b, d_r, d_p,
           d_r1);
  spmvAp(handle, descr, d_val, d_ptr, d_col, d_p, d_Ap);

  cudaEvent_t start, stop, LUSTart, LUStop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&LUSTart);
  cudaEventCreate(&LUStop);
  cudaEventRecord(start, 0);
  // cout << "residual:"<<residual() << endl;
  double residual;
  cublasDnrm2(handle2, num_rows, d_r, 1, &residual);
  cout << "residual:" << residual * residual << endl;
  while (iteration < MAX_ITERATION && residual * residual > threshold) {

    calAlpha(handle2, d_r, d_Ap);
    calXAndR(handle2, d_p, d_Ap, d_r, d_x);
    cudaMemcpy(r, d_r, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);
    /*serial*/
    cudaEventRecord(LUSTart, 0);
    LU();
    cudaEventRecord(LUStop, 0);
    cudaEventSynchronize(LUStop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, LUSTart, LUStop);
    serialLU += elapsedTime;
    cudaMemcpy(d_r1, r1, sizeof(double) * num_rows, cudaMemcpyHostToDevice);
    spmvAr(handle, descr, d_val, d_ptr, d_col, d_r1, d_Ar);
    calBeta(handle2, d_Ar, d_Ap, d_beta);
    calPAndAp(handle2, d_p, d_Ap, d_r1, d_Ar, p_temp, Ap_temp);
    cublasDnrm2(handle2, num_rows, d_r, 1, &residual);
    iteration++;
    cout << "residual:" << residual * residual << endl;
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cout << "***********" << endl;
  cout << "iteration:" << iteration << endl;
  cout << "residual:" << residual * residual << endl;
  cout << "time:" << elapsedTime / 1000 << "s" << endl;
  cout << "LU time:" << serialLU / 1000 << "s" << endl;

  memFree();
  cudaFree(d_ptr);
  cudaFree(d_val);
  cudaFree(d_col);
  cudaFree(d_x);
  cudaFree(d_b);
  cudaFree(d_p);
  cudaFree(d_Ap);
  cudaFree(d_r);
  cudaFree(d_Ar);
  cudaFree(d_beta);
  cudaFree(p_temp);
  cudaFree(Ap_temp);

  cusparseDestroyMatDescr(descr);
  cusparseDestroy(handle);
  cublasDestroy(handle2);
  return 0;
}

/*
 *ilu0分解，分为单位下三角矩阵和上三角矩阵，合并保存到原矩阵空间
 */
void ilu0() {
  int i, j, k;
  int start, end;
  int startk, endk;
  for (i = 1; i < num_rows; i++) {
    start = ptr[i];
    end = ptr[i + 1];
    for (k = start; col[k] < i; k++) {
      val_m[k] /= val_m[mid_index[col[k]]];

      startk = ptr[col[k]];   // col k 行 start
      endk = ptr[col[k] + 1]; // col k 行 end
      for (j = k + 1; j < end; j++) {
        while (col[startk] < col[j] &&
               startk < endk) { // col k行找j列，两种情况退出 找到或没找到
          startk++;
        }
        if (col[startk] == col[j]) { //没找到跳过
          val_m[j] -= val_m[k] * val_m[startk];
        }
      }
    }
  }
}

/*
 *三角回代
 */
__host__ void LU() {
  double temp;
  for (int i = 0; i < num_rows; i++) {
    int start = ptr[i];
    temp = 0;
    for (int j = start; col[j] < i; j++) {
      temp += val_m[j] * y[col[j]];
    }

    y[i] = r[i] - temp;
  }

  for (int i = num_rows - 1; i >= 0; i--) {
    int end = ptr[i + 1];
    temp = 0;
    for (int j = end - 1; col[j] > i; j--) {
      temp += val_m[j] * r1[col[j]];
    }
    r1[i] = (y[i] - temp) / val_m[mid_index[i]];
  }
}

__host__ void memFree() {
  free(ptr);
  free(col);
  free(val);
  cudaFreeHost(x);
  free(b);
  cudaFreeHost(Ap);
  cudaFreeHost(Ar);
  cudaFreeHost(p);
  cudaFreeHost(r);
  cudaFreeHost(r1);
  cudaFreeHost(beta);
}

__host__ void calPAndAp(cublasHandle_t handle2, double *d_p, double *d_Ap,
                        double *d_r1, double *d_Ar, double *p_temp,
                        double *Ap_temp) {
  cublasDcopy(handle2, num_rows, d_r1, 1, p_temp, 1);
  cublasDcopy(handle2, num_rows, d_Ar, 1, Ap_temp, 1);
  for (int j = 0; j < num_k; j++) {
    cublasDaxpy(handle2, num_rows, &beta[j], d_p + j * num_rows, 1, p_temp, 1);
    cublasDaxpy(handle2, num_rows, &beta[j], d_Ap + j * num_rows, 1, Ap_temp,
                1);
  }

  cublasDcopy(handle2, num_rows, p_temp, 1,
              d_p + (iteration % num_k) * num_rows, 1);
  cublasDcopy(handle2, num_rows, Ap_temp, 1,
              d_Ap + (iteration % num_k) * num_rows, 1);
}

__host__ void calBeta(cublasHandle_t handle2, double *d_Ar, double *d_Ap,
                      double *d_beta) {
  int start = iteration > num_k
                  ? ((iteration / num_k) - 1) * num_k + iteration % num_k
                  : 0;
  for (int j = start; j < iteration; j++) {
    double s, m;
    cublasDdot(handle2, num_rows, d_Ar, 1, d_Ap + (j % num_k) * num_rows, 1,
               &s);
    cublasDdot(handle2, num_rows, d_Ap + (j % num_k) * num_rows, 1,
               d_Ap + (j % num_k) * num_rows, 1, &m);
    beta[j % num_k] = -1 * (s / m);
  }
}

__host__ void spmvAr(cusparseHandle_t handle, cusparseMatDescr_t descr,
                     double *d_val, int *d_ptr, int *d_col, double *d_r1,
                     double *d_Ar) {

  double d1 = 1, d2 = 0;
  cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rows,
                 num_elements, &d1, descr, d_val, d_ptr, d_col, d_r1, &d2,
                 d_Ar);
}

__host__ void calXAndR(cublasHandle_t handle2, double *d_p, double *d_Ap,
                       double *d_r, double *d_x) {
  double alpha2 = -alpha;
  cublasDaxpy(handle2, num_rows, &alpha,
              d_p + ((iteration - 1) % num_k) * num_rows, 1, d_x, 1);
  cublasDaxpy(handle2, num_rows, &alpha2,
              d_Ap + ((iteration - 1) % num_k) * num_rows, 1, d_r, 1);
}

__host__ void calAlpha(cublasHandle_t handle2, double *d_r, double *d_Ap) {
  double s, m;
  cublasDdot(handle2, num_rows, d_r, 1,
             d_Ap + ((iteration - 1) % num_k) * num_rows, 1, &s);
  cublasDdot(handle2, num_rows, d_Ap + ((iteration - 1) % num_k) * num_rows, 1,
             d_Ap + ((iteration - 1) % num_k) * num_rows, 1, &m);
  alpha = s / m;
}

__host__ void spmvAp(cusparseHandle_t handle, cusparseMatDescr_t descr,
                     double *d_val, int *d_ptr, int *d_col, double *d_p,
                     double *d_Ap) {
  double d1 = 1, d2 = 0;
  cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rows,
                 num_elements, &d1, descr, d_val, d_ptr, d_col, d_p, &d2, d_Ap);
}

__host__ void initData(cusparseHandle_t handle, cublasHandle_t handle2,
                       cusparseMatDescr_t descr, double *d_val, int *d_ptr,
                       int *d_col, double *d_x, double *d_b, double *d_r,
                       double *d_p, double *d_r1) {
  // int blockNum = (num_rows % THREADS_PER_BLOCK) > 0 ? (num_rows /
  // THREADS_PER_BLOCK) + 1 : (num_rows / THREADS_PER_BLOCK);
  double d2 = 1, d3 = -1;
  cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rows,
                 num_elements, &d3, descr, d_val, d_ptr, d_col, d_x, &d2, d_b);
  cublasDcopy(handle2, num_rows, d_b, 1, d_r, 1);
  cudaMemcpy(r, d_r, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);
  LU();
  cudaMemcpy(d_r1, r1, sizeof(double) * num_rows, cudaMemcpyHostToDevice);
  cublasDcopy(handle2, num_rows, d_r1, 1, d_p, 1);
  /*
  cudaMemcpy(r, d_r, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);
  cudaMemcpy(p, d_p, sizeof(double) * num_rows * num_k, cudaMemcpyDeviceToHost);
  cout << r[3] << endl;
  cout << p[3] << endl;
  */
}

void readMatrix() {
  A = (double *)malloc(sizeof(double) * size * NEIGHBOR);
  b = (double *)malloc(sizeof(double) * size);
  x = (double *)malloc(sizeof(double) * size);
  f1 = fopen("case/A", "rb");
  fread(A, sizeof(double), size * NEIGHBOR, f1);
  fclose(f1);
  f2 = fopen("case/b", "rb");
  fread(b, sizeof(double), size, f2);
  fclose(f2);
  f3 = fopen("case/x0", "rb");
  fread(x, sizeof(double), size, f3);
  fclose(f3);
  ptr = (int *)malloc(sizeof(int) * (size + 1));
  col = (int *)malloc(sizeof(int) * size * NEIGHBOR);
  val = (double *)malloc(sizeof(double) * size * NEIGHBOR);
  mid_index = (int *)malloc(sizeof(int) * num_rows); //对角线元素index

  Ap = (double *)malloc(sizeof(double) * num_rows * num_k);
  Ar = (double *)malloc(sizeof(double) * num_rows);
  r = (double *)malloc(sizeof(double) * num_rows);
  r1 = (double *)malloc(sizeof(double) * num_rows); // ilu variable
  y = (double *)malloc(sizeof(double) * num_rows);  // ilu variable
  p = (double *)malloc(sizeof(double) * num_rows * num_k);
  if (p == NULL)
    printf("memory error\n");
  beta = (double *)malloc(sizeof(double) * num_k);
  memset(Ap, 0, num_rows * num_k * sizeof(double));
  memset(p, 0, num_rows * num_k * sizeof(double));
  memset(beta, 0, num_k * sizeof(double));

  ptr[0] = 0;
  int index = 0;

  int i, j, k; // nx,ny,nz
  //#pragma omp parallel private(i,j,k,index)
  {
    //#pragma omp for
    for (i = 0; i < nx; i++)
      for (j = 0; j < ny; j++)
        for (k = 0; k < nz; k++) {
          col[index] = IDX(i, j, k);
          val[index++] = A[IDXA(i, j, k, 0)];

          if (i > 0)
            col[index] = IDX(i - 1, j, k);
          else
            col[index] = IDX(nx - 1, j, k);
          val[index++] = A[IDXA(i, j, k, 1)];

          if (i < nx - 1)
            col[index] = IDX(i + 1, j, k);
          else
            col[index] = IDX(0, j, k);
          val[index++] = A[IDXA(i, j, k, 2)];

          if (j > 0)
            col[index] = IDX(i, j - 1, k);
          else
            col[index] = IDX((i + nx / 2) % nx, j, k);
          val[index++] = A[IDXA(i, j, k, 3)];

          if (j < ny - 1)
            col[index] = IDX(i, j + 1, k);
          else
            col[index] = IDX((i + nx / 2) % nx, j, k);
          val[index++] = A[IDXA(i, j, k, 4)];

          if (i < nx - 1 && j < ny - 1)
            col[index] = IDX(i + 1, j + 1, k);
          else if (i < nx - 1 && j == ny - 1)
            col[index] = IDX((i + 1 + nx / 2) % nx, j, k);
          else if (i == nx - 1 && j < ny - 1)
            col[index] = IDX(0, j + 1, k);
          else
            col[index] = IDX(nx / 2, j, k);
          val[index++] = A[IDXA(i, j, k, 5)];

          if (i < nx - 1 && j > 0)
            col[index] = IDX(i + 1, j - 1, k);
          else if (i < nx - 1 && j == 0)
            col[index] = IDX((i + 1 + nx / 2) % nx, j, k);
          else if (i == nx - 1 && j > 0)
            col[index] = IDX(0, j - 1, k);
          else
            col[index] = IDX(nx / 2, j, k);
          val[index++] = A[IDXA(i, j, k, 6)];

          if (i > 0 && j > 0)
            col[index] = IDX(i - 1, j - 1, k);
          else if (i > 0 && j == 0)
            col[index] = IDX((i - 1 + nx / 2) % nx, j, k);
          else if (i == 0 && j > 0)
            col[index] = IDX(nx - 1, j - 1, k);
          else
            col[index] = IDX((nx - 1 + nx / 2) % nx, j, k);
          val[index++] = A[IDXA(i, j, k, 7)];

          if (i > 0 && j < ny - 1)
            col[index] = IDX(i - 1, j + 1, k);
          else if (i > 0 && j == ny - 1)
            col[index] = IDX((i - 1 + nx / 2) % nx, j, k);
          else if (i == 0 && j < ny - 1)
            col[index] = IDX(nx - 1, j + 1, k);
          else
            col[index] = IDX((nx - 1 + nx / 2) % nx, j, k);
          val[index++] = A[IDXA(i, j, k, 8)];

          if (k > 0)
            col[index] = IDX(i, j, k - 1), val[index++] = A[IDXA(i, j, k, 9)];

          if (k > 0 && i > 0)
            col[index] = IDX(i - 1, j, k - 1),
            val[index++] = A[IDXA(i, j, k, 10)];
          else if (k > 0 && i == 0)
            col[index] = IDX(nx - 1, j, k - 1),
            val[index++] = A[IDXA(i, j, k, 10)];

          if (k > 0 && i < nx - 1)
            col[index] = IDX(i + 1, j, k - 1),
            val[index++] = A[IDXA(i, j, k, 11)];
          else if (k > 0 && i == nx - 1)
            col[index] = IDX(0, j, k - 1), val[index++] = A[IDXA(i, j, k, 11)];

          if (k > 0 && j > 0)
            col[index] = IDX(i, j - 1, k - 1),
            val[index++] = A[IDXA(i, j, k, 12)];
          else if (k > 0 && j == 0)
            col[index] = IDX((i + nx / 2) % nx, j, k - 1),
            val[index++] = A[IDXA(i, j, k, 12)];

          if (k > 0 && j < ny - 1)
            col[index] = IDX(i, j + 1, k - 1),
            val[index++] = A[IDXA(i, j, k, 13)];
          else if (k > 0 && j == ny - 1)
            col[index] = IDX((i + nx / 2) % nx, j, k - 1),
            val[index++] = A[IDXA(i, j, k, 13)];

          if (k < nz - 1)
            col[index] = IDX(i, j, k + 1), val[index++] = A[IDXA(i, j, k, 14)];

          if (k < nz - 1 && i > 0)
            col[index] = IDX(i - 1, j, k + 1),
            val[index++] = A[IDXA(i, j, k, 15)];
          else if (k < nz - 1 && i == 0)
            col[index] = IDX(nx - 1, j, k + 1),
            val[index++] = A[IDXA(i, j, k, 15)];

          if (k < nz - 1 && i < nx - 1)
            col[index] = IDX(i + 1, j, k + 1),
            val[index++] = A[IDXA(i, j, k, 16)];
          else if (k < nz - 1 && i == nx - 1)
            col[index] = IDX(0, j, k + 1), val[index++] = A[IDXA(i, j, k, 16)];

          if (k < nz - 1 && j > 0)
            col[index] = IDX(i, j - 1, k + 1),
            val[index++] = A[IDXA(i, j, k, 17)];
          else if (k < nz - 1 && j == 0)
            col[index] = IDX((i + nx / 2) % nx, j, k + 1),
            val[index++] = A[IDXA(i, j, k, 17)];

          if (k < nz - 1 && j < ny - 1)
            col[index] = IDX(i, j + 1, k + 1),
            val[index++] = A[IDXA(i, j, k, 18)];
          else if (k < nz - 1 && j == ny - 1)
            col[index] = IDX((i + nx / 2) % nx, j, k + 1),
            val[index++] = A[IDXA(i, j, k, 18)];

          ptr[IDX(i, j, k) + 1] = index;
        }
  }

  // sort
  for (int i = 0; i < num_rows; i++) {
    int start = ptr[i];
    int end = ptr[i + 1];
    double temp_d;
    int temp_i;
    for (int j = start; j < end - 1; j++) {

      for (int k = start; k < end - 1 - j + start; k++) {

        if (col[k] > col[k + 1]) {
          temp_d = val[k + 1];
          temp_i = col[k + 1];
          val[k + 1] = val[k];
          col[k + 1] = col[k];
          val[k] = temp_d;
          col[k] = temp_i;
        }
      }
    }
  }
  //找每行对角中间元素的index
  for (int i = 0; i < num_rows; i++) {
    int start = ptr[i];
    int end = ptr[i + 1];
    for (int j = start; j < end; j++) {
      if (col[j] == i)
        mid_index[i] = j;
    }
  }

  free(A); // free space
  num_elements = index;
  /*csr ilu0*/
  val_m = (double *)malloc(sizeof(double) * size * NEIGHBOR);
  memcpy(val_m, val, sizeof(double) * size * NEIGHBOR); // cpy val_m
}
