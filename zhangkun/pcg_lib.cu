#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cusparse.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>

using namespace std;
#define MAX_ITERATION 200
#define THRESHOLD 1e-4
#define THREADS_PER_BLOCK 512
// initial data
int num_rows;
int num_col;
int num_elements;

// coo format
int *row;
int *column;
double *value;
// temp ptr_count  store the nnz of each row
int *ptr_cnt;
// csr format
int *ptr;
int *col;
double *val;
// csr ilu0
double *val_m;
int *mid_index;

double *b;
double *x;

double threshold = THRESHOLD;
int iteration = 0;
double res = 0;

double *r0;
double *r1;
double *z0;
double *z1;
double *y; // temp variable
double beta;
double alpha;
double *p;
double *Ap;

string field = "";
string symmetry = "";

string filename = "offshore";

int main(int argc, char **argv) {
  if (argc > 1)
    filename = argv[1];

  __host__ void readMatrix();
  __host__ void csr(); // change format  coo2csr
  __host__ void ilu0();
  __host__ void LU(double *r, double *z);

  __host__ void initData(cusparseHandle_t handle, cublasHandle_t handle2,
                         cusparseMatDescr_t descr, double *d_val, int *d_ptr,
                         int *d_col, double *d_x, double *d_b, double *d_r,
                         double *r, double *d_p, double *d_z, double *z);

  __host__ void spmvAp(cusparseHandle_t handle, cusparseMatDescr_t descr,
                       double *d_val, int *d_ptr, int *d_col, double *d_p,
                       double *d_Ap);
  __host__ void calAlpha(cublasHandle_t handle2, double *d_r0, double *d_z0,
                         double *d_Ap, double *d_p);
  __host__ void calXAndR(cublasHandle_t handle2, double *d_p, double *d_Ap,
                         double *d_r0, double *d_r1, double *d_x);
  __host__ void calBeta(cublasHandle_t handle2, double *d_r1, double *d_z1,
                        double *d_r0, double *d_z0);
  // time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  readMatrix();
  cout << "read finish" << endl;
  csr();
  ilu0();

  /*device variable*/
  int *d_ptr;
  int *d_col;
  double *d_val;
  double *d_b;
  double *d_x;
  // iterative variable
  double *d_r0;
  double *d_r1;
  double *d_z0;
  double *d_z1;
  double *d_p;
  double *d_Ap;

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

  cudaMemcpy(d_ptr, ptr, sizeof(int) * (num_rows + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col, col, sizeof(int) * (num_elements), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, val, sizeof(double) * (num_elements),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ap, Ap, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, p, sizeof(double) * (num_rows), cudaMemcpyHostToDevice);

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

  cudaEventRecord(start, 0);

  initData(handle, handle2, descr, d_val, d_ptr, d_col, d_x, d_b, d_r0, r0, d_p,
           d_z0, z0);
  cublasDnrm2(handle2, num_rows, d_r0, 1, &res);
  cout << "res:" << scientific << setprecision(3) << res << endl;
  while (iteration < MAX_ITERATION) {
    spmvAp(handle, descr, d_val, d_ptr, d_col, d_p, d_Ap);
    calAlpha(handle2, d_r0, d_z0, d_Ap, d_p);
    calXAndR(handle2, d_p, d_Ap, d_r0, d_r1, d_x);
    cudaMemcpy(r1, d_r1, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);
    LU(r1, z1);
    cudaMemcpy(d_z1, z1, sizeof(double) * num_rows, cudaMemcpyHostToDevice);
    calBeta(handle2, d_r1, d_z1, d_r0, d_z0);
    cublasDaxpy(handle2, num_rows, &beta, d_p, 1, d_z1, 1);
    cublasDcopy(handle2, num_rows, d_z1, 1, d_p, 1);
    cublasDnrm2(handle2, num_rows, d_r0, 1, &res);
    iteration++;
    cout << "ite:" << iteration;
    cout << "  res:" << scientific << setprecision(3) << res << endl;
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cout << "***********" << endl;
  cout << "ite:" << iteration;
  cout << "  res:" << scientific << setprecision(3) << res << endl;
  cout << fixed << "time:" << elapsedTime / 1000 << "s" << endl;
}

void csr() {
  /*nnz of each row*/
  for (int i = 0; i < num_elements; i++) {
    ptr_cnt[row[i]]++;
  }
  /*scan ptr*/
  for (int i = 0; i < num_rows; i++) {
    ptr[i + 1] = ptr_cnt[i] + ptr[i];
  }
  memset(ptr_cnt, 0, num_rows * sizeof(int)); // set 0
  int offset;                                 // start of each row
  for (int i = 0; i < num_elements; i++) {
    offset = ptr[row[i]] + ptr_cnt[row[i]];
    col[offset] = column[i];
    val[offset] = value[i];
    ptr_cnt[row[i]]++;
  }

  /*sort each row*/
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

  // find mid_index
  for (int i = 0; i < num_rows; i++) {
    int start = ptr[i];
    int end = ptr[i + 1];
    for (int j = start; j < end; j++) {
      if (col[j] == i)
        mid_index[i] = j;
    }
  }
  // cpy val_m
  memcpy(val_m, val, sizeof(double) * num_elements);
  memset(Ap, 0, num_rows * sizeof(double));
  memset(p, 0, num_rows * sizeof(double));
}

/*init ilu0*/
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

/*SpTRSV*/
void LU(double *r, double *z) {
  double temp;
  for (int i = 0; i < num_rows; i++) {
    int start = ptr[i];
    // int end = ptr[i + 1];
    temp = 0;
    for (int j = start; col[j] < i; j++) {
      temp += val_m[j] * y[col[j]];
    }

    y[i] = r[i] - temp;
  }

  // for(int i=0;i<100;i++)
  // 	printf("%.3e ",y[i]);

  for (int i = num_rows - 1; i >= 0; i--) {
    // int start = ptr[i];
    int end = ptr[i + 1];
    temp = 0;
    for (int j = end - 1; col[j] > i; j--) {
      temp += val_m[j] * z[col[j]];
    }

    z[i] = (y[i] - temp) / val_m[mid_index[i]];
  }
}

void readMatrix() {
  // cout << "input matrix filename!" << endl;
  // cin>>filename;
  // filename = "t";
  ifstream fin;
  // string base = "D:\\doc\\GRAPES\\code\\SpMV\\matrix\\";
  string base = "matrix/";
  string name = base + filename + ".mtx";
  fin.open(name.c_str());

  for (int i = 0; i < 4; i++)
    fin >> field;
  fin >> symmetry;

  fin.ignore(2048, '\n');
  while (fin.peek() == '%') // ignore row
  {
    fin.ignore(2048, '\n');
  }

  fin >> num_rows >> num_col >> num_elements;
  if (symmetry == "symmetric") {
    num_elements = num_elements * 2 - num_rows;
    cout << "num_rows:" << num_rows << "  num_cols:" << num_col
         << "  num_elements(symmetric):" << num_elements << endl;
  } else
    cout << "num_rows:" << num_rows << "  num_cols:" << num_col
         << "  num_elements(general):" << num_elements << endl;
  // coo
  row = new int[num_elements];
  column = new int[num_elements];
  value = new double[num_elements];
  // temp ptr_count
  ptr_cnt = new int[num_rows];
  // csr
  ptr = new int[num_rows + 1];
  col = new int[num_elements];
  val = new double[num_elements];

  // csr ilu0
  val_m = new double[num_elements];
  mid_index = new int[num_rows];

  // CSR    first element  must be zero!!!!!
  ptr[0] = 0;

  if (field == "real" || field == "integer") // row col value
  {
    int sign = 1;
    if (symmetry == "symmetric")
      for (int i = 0; i < (num_elements + num_rows) / 2; i++) {
        int m = 0, n = 0;
        fin >> m >> n >> value[i];
        if (m != n) {
          row[i] = m - 1;
          column[i] = n - 1;

          value[num_elements - sign] = value[i];
          row[num_elements - sign] = n - 1;
          column[num_elements - sign] = m - 1;
          sign++;
        } else {
          row[i] = m - 1;
          column[i] = n - 1;
        }
      }
    else
      for (int i = 0; i < num_elements; i++) {
        int m = 0, n = 0;
        fin >> m >> n >> value[i];
        row[i] = m - 1;
        column[i] = n - 1;
      }
  } else if (field == "pattern") // non-value   default 1
  {
    int sign = 1;
    if (symmetry == "symmetric")
      for (int i = 0; i < (num_elements + num_rows) / 2; i++) {
        int m = 0, n = 0;
        fin >> m >> n;
        if (m != n) {
          value[i] = 1;
          row[i] = m - 1;
          column[i] = n - 1;

          value[num_elements - sign] = 1;
          row[num_elements - sign] = n - 1;
          column[num_elements - sign] = m - 1;

          sign++;
        } else {
          value[i] = 1;
          row[i] = m - 1;
          column[i] = n - 1;
        }
      }

    else
      for (int i = 0; i < num_elements; i++) {
        int m = 0, n = 0;
        fin >> m >> n;
        value[i] = 1;
        row[i] = m - 1;
        column[i] = n - 1;
      }
  }
  fin.close();
  // memory allocate
  r0 = new double[num_rows];
  r1 = new double[num_rows];
  z0 = new double[num_rows];
  z1 = new double[num_rows];
  y = new double[num_rows];
  Ap = new double[num_rows];
  p = new double[num_rows];
  b = new double[num_rows];
  x = new double[num_rows];

  for (int i = 0; i < num_rows; i++) {
    x[i] = 1;
    b[i] = 1;
  }
  memset(ptr_cnt, 0, num_rows * sizeof(int)); // set 0
}

__host__ void initData(cusparseHandle_t handle, cublasHandle_t handle2,
                       cusparseMatDescr_t descr, double *d_val, int *d_ptr,
                       int *d_col, double *d_x, double *d_b, double *d_r,
                       double *r, double *d_p, double *d_z, double *z) {

  double d2 = 1, d3 = -1;
  cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rows,
                 num_elements, &d3, descr, d_val, d_ptr, d_col, d_x, &d2, d_b);
  cublasDcopy(handle2, num_rows, d_b, 1, d_r, 1);
  cudaMemcpy(r, d_r, sizeof(double) * num_rows, cudaMemcpyDeviceToHost);
  LU(r, z);
  cudaMemcpy(d_z, z, sizeof(double) * num_rows, cudaMemcpyHostToDevice);
  cublasDcopy(handle2, num_rows, d_z, 1, d_p, 1);
}

__host__ void spmvAp(cusparseHandle_t handle, cusparseMatDescr_t descr,
                     double *d_val, int *d_ptr, int *d_col, double *d_p,
                     double *d_Ap) {
  double d1 = 1, d2 = 0;
  cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_rows,
                 num_elements, &d1, descr, d_val, d_ptr, d_col, d_p, &d2, d_Ap);
}

__host__ void calAlpha(cublasHandle_t handle2, double *d_r0, double *d_z0,
                       double *d_Ap, double *d_p) {
  double s, m;
  cublasDdot(handle2, num_rows, d_r0, 1, d_z0, 1, &s);
  cublasDdot(handle2, num_rows, d_Ap, 1, d_p, 1, &m);
  alpha = s / m;
}
__host__ void calXAndR(cublasHandle_t handle2, double *d_p, double *d_Ap,
                       double *d_r0, double *d_r1, double *d_x) {
  cublasDcopy(handle2, num_rows, d_r0, 1, d_r1, 1);
  double alpha2 = -alpha;
  cublasDaxpy(handle2, num_rows, &alpha, d_p, 1, d_x, 1);
  cublasDaxpy(handle2, num_rows, &alpha2, d_Ap, 1, d_r1, 1);
}

__host__ void calBeta(cublasHandle_t handle2, double *d_r1, double *d_z1,
                      double *d_r0, double *d_z0) {
  double s, m;
  cublasDdot(handle2, num_rows, d_r1, 1, d_z1, 1, &s);
  cublasDdot(handle2, num_rows, d_r0, 1, d_z0, 1, &m);
  beta = s / m;

  cublasDcopy(handle2, num_rows, d_r1, 1, d_r0, 1);
  cublasDcopy(handle2, num_rows, d_z1, 1, d_z0, 1);
}