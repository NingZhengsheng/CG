#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <iomanip>
#include <omp.h>

using namespace std;
#define MAX_ITERATION 500
#define THRESHOLD 1e-4

//initial data
int num_rows;
int num_col;
int num_elements;

//coo format
int *row;
int *column;
double *value;
//temp ptr_count  store the nnz of each row
int *ptr_cnt;
//csr format
int *ptr;
int *col;
double *val;
//csr ilu0
double *val_m;
int *mid_index;
/*jacobi = 1/dia(A)*/
double *jacobi;

double *b;
double *x;

double threshold = THRESHOLD;
int iteration = 0;
double norm = 0;

double *r0;
double *r1;
double *z0;
double *z1;
double *y; //temp variable
double beta;
double alpha;
double *p;
double *Ap;

string field = "";
string symmetry = "";

string filename = "Dubcova2";

int main(int argc, char **argv)
{
    if (argc > 1)
        filename = argv[1];
    void readMatrix();
    void csr();                                   //change format  coo2csr
    void ilu0();                                  //init ilu0 matrix
    void LU(double *r, double *z);                //sptrsv
    void Jacobi(double *r, double *z);            //z=(p^-1)r
    double getResidual_csr();                     //compute norm
    void getR0_csr();                             //init r0
    void matrixMultiA_csr(double *vec);           //spmv
    double multiVector(double *a, double *b);     //dot
    void numMultiVector(double num, double *vec); //scalar
    void replaceVector(double *a, double *b);     //copy a=b

    clock_t start, end;
    //omp_set_num_threads(12);
    //cin>>filename;

    readMatrix();
    cout << "read finish" << endl;

    start = clock();
    csr();
    end = clock();
    cout << "trans time:" << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;

    start = clock();
    ilu0();
    end = clock();
    cout << "ilu0 time:" << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;

    getR0_csr();

    LU(r0, z0);
    //Jacobi(z0,z0);
    start = clock();

    for (int i = 0; i < num_rows; i++)
        p[i] = z0[i]; // p0

    norm = getResidual_csr();
    while (iteration < MAX_ITERATION && norm > threshold) //02
    {

        matrixMultiA_csr(p);
        alpha = multiVector(r0, z0) / multiVector(Ap, p);
        for (int i = 0; i < num_rows; i++)
            x[i] = x[i] + alpha * p[i];
        for (int i = 0; i < num_rows; i++)
            r1[i] = r0[i] - alpha * Ap[i];

        LU(r1, z1);
        //Jacobi(z1,z1);
        beta = multiVector(r1, z1) / multiVector(r0, z0);
        for (int i = 0; i < num_rows; i++)
            p[i] = z1[i] + beta * p[i];
        replaceVector(r0, r1);
        replaceVector(z0, z1);
        iteration++;
        norm = getResidual_csr(); 
        cout << "ite:" << iteration;
        cout << "  res:" << scientific << setprecision(3) << norm << endl;
    }

    end = clock();

    cout << "***********************" << endl;
    cout << "num of iteration:" << iteration << endl;
    cout << "residual:" << scientific << setprecision(3) << getResidual_csr() << endl;
    cout << "time:" << fixed << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;

    getR0_csr();
    cout << "real residual:" << scientific << setprecision(3) << getResidual_csr() << endl;

    return 0;
}

void readMatrix()
{
    //cout << "input matrix filename!" << endl;
    //cin>>filename;
    //filename = "t";
    ifstream fin;
    //string base = "D:\\doc\\GRAPES\\code\\SpMV\\matrix\\";
    string base = "matrix/";
    string name = base + filename + ".mtx";
    fin.open(name.c_str());

    for (int i = 0; i < 4; i++)
        fin >> field;
    fin >> symmetry;

    fin.ignore(2048, '\n');
    while (fin.peek() == '%') //ignore row
    {
        fin.ignore(2048, '\n');
    }

    fin >> num_rows >> num_col >> num_elements;
    if (symmetry == "symmetric")
    {
        num_elements = num_elements * 2 - num_rows;
        cout << "num_rows:" << num_rows << "  num_cols:" << num_col << "  num_elements(symmetric):" << num_elements << endl;
    }
    else
        cout << "num_rows:" << num_rows << "  num_cols:" << num_col << "  num_elements(general):" << num_elements << endl;
    //coo
    row = new int[num_elements];
    column = new int[num_elements];
    value = new double[num_elements];
    //temp ptr_count
    ptr_cnt = new int[num_rows];
    //csr
    ptr = new int[num_rows + 1];
    col = new int[num_elements];
    val = new double[num_elements];

    //csr ilu0
    val_m = new double[num_elements];
    mid_index = new int[num_rows];
    //jacobi
    jacobi = new double[num_rows];

    // CSR    first element  must be zero!!!!!
    ptr[0] = 0;

    if (field == "real" || field == "integer") // row col value
    {
        int sign = 1;
        if (symmetry == "symmetric")
            for (int i = 0; i < (num_elements + num_rows) / 2; i++)
            {
                int m = 0, n = 0;
                fin >> m >> n >> value[i];
                if (m != n)
                {
                    row[i] = m - 1;
                    column[i] = n - 1;

                    value[num_elements - sign] = value[i];
                    row[num_elements - sign] = n - 1;
                    column[num_elements - sign] = m - 1;
                    sign++;
                }
                else
                {
                    row[i] = m - 1;
                    column[i] = n - 1;
                }
            }
        else
            for (int i = 0; i < num_elements; i++)
            {
                int m = 0, n = 0;
                fin >> m >> n >> value[i];
                row[i] = m - 1;
                column[i] = n - 1;
            }
    }
    else if (field == "pattern") //non-value   default 1
    {
        int sign = 1;
        if (symmetry == "symmetric")
            for (int i = 0; i < (num_elements + num_rows) / 2; i++)
            {
                int m = 0, n = 0;
                fin >> m >> n;
                if (m != n)
                {
                    value[i] = 1;
                    row[i] = m - 1;
                    column[i] = n - 1;

                    value[num_elements - sign] = 1;
                    row[num_elements - sign] = n - 1;
                    column[num_elements - sign] = m - 1;

                    sign++;
                }
                else
                {
                    value[i] = 1;
                    row[i] = m - 1;
                    column[i] = n - 1;
                }
            }

        else
            for (int i = 0; i < num_elements; i++)
            {
                int m = 0, n = 0;
                fin >> m >> n;
                value[i] = 1;
                row[i] = m - 1;
                column[i] = n - 1;
            }
    }
    fin.close();
    //memory allocate
    r0 = new double[num_rows];
    r1 = new double[num_rows];
    z0 = new double[num_rows];
    z1 = new double[num_rows];
    y = new double[num_rows];
    Ap = new double[num_rows];
    p = new double[num_rows];
    b = new double[num_rows];
    x = new double[num_rows];

    for (int i = 0; i < num_rows; i++)
    {
        x[i] = 1;
        b[i] = 1;
    }
    memset(ptr_cnt, 0, num_rows * sizeof(int)); //set 0
}

void csr()
{
    /*nnz of each row*/
    for (int i = 0; i < num_elements; i++)
    {
        ptr_cnt[row[i]]++;
    }
    /*scan ptr*/
    for (int i = 0; i < num_rows; i++)
    {
        ptr[i + 1] = ptr_cnt[i] + ptr[i];
    }
    memset(ptr_cnt, 0, num_rows * sizeof(int)); //set 0
    int offset;                                 //start of each row
    for (int i = 0; i < num_elements; i++)
    {
        offset = ptr[row[i]] + ptr_cnt[row[i]];
        col[offset] = column[i];
        val[offset] = value[i];
        ptr_cnt[row[i]]++;
    }

    /*sort each row*/
    for (int i = 0; i < num_rows; i++)
    {
        int start = ptr[i];
        int end = ptr[i + 1];
        double temp_d;
        int temp_i;
        for (int j = start; j < end - 1; j++)
        {

            for (int k = start; k < end - 1 - j + start; k++)
            {

                if (col[k] > col[k + 1])
                {
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

    //find mid_index
    for (int i = 0; i < num_rows; i++)
    {
        int start = ptr[i];
        int end = ptr[i + 1];
        for (int j = start; j < end; j++)
        {
            if (col[j] == i)
            {
                mid_index[i] = j;
                jacobi[i] = 1 / val[j];
            }
        }
    }
    //cpy val_m
    memcpy(val_m, val, sizeof(double) * num_elements);
}

/*init ilu0*/
void ilu0()
{
    int i, j, k;
    int start, end;
    int startk, endk;
    for (i = 1; i < num_rows; i++)
    {
        start = ptr[i];
        end = ptr[i + 1];
        for (k = start; col[k] < i; k++)
        {
            val_m[k] /= val_m[mid_index[col[k]]];

            startk = ptr[col[k]];   //col k 行 start
            endk = ptr[col[k] + 1]; //col k 行 end
            for (j = k + 1; j < end; j++)
            {
                while (col[startk] < col[j] && startk < endk)
                { //col k行找j列，两种情况退出 找到或没找到
                    startk++;
                }
                if (col[startk] == col[j])
                { //没找到跳过
                    val_m[j] -= val_m[k] * val_m[startk];
                }
            }
        }
    }
}

/*jacobi preconditioner*/
void Jacobi(double *r, double *z)
{
    for (int i = 0; i < num_rows; i++)
    {
        z[i] = jacobi[i] * r[i];
    }
}

/*SpTRSV*/
void LU(double *r, double *z)
{
    double temp;
    for (int i = 0; i < num_rows; i++)
    {
        int start = ptr[i];
        int end = ptr[i + 1];
        temp = 0;
        for (int j = start; col[j] < i; j++)
        {
            temp += val_m[j] * y[col[j]];
        }

        y[i] = r[i] - temp;
    }

    // for(int i=0;i<100;i++)
    // 	printf("%.3e ",y[i]);

    for (int i = num_rows - 1; i >= 0; i--)
    {
        int start = ptr[i];
        int end = ptr[i + 1];
        temp = 0;
        for (int j = end - 1; col[j] > i; j--)
        {
            temp += val_m[j] * z[col[j]];
        }

        z[i] = (y[i] - temp) / val_m[mid_index[i]];
    }
}

void getR0_csr()
{
    for (int row = 0; row < num_rows; row++)
    {
        double sum_of_row = 0;
        int row_start = ptr[row];
        int row_end = ptr[row + 1];
        for (int i = row_start; i < row_end; i++)
        {
            sum_of_row += val[i] * x[col[i]];
        }
        r0[row] = (b[row] - sum_of_row);
        r1[row] = (b[row] - sum_of_row);
    }
}
void matrixMultiA_csr(double *vec)
{
    double sum_of_row;
    for (int row = 0; row < num_rows; row++)
    {
        sum_of_row = 0;
        int row_start = ptr[row];
        int row_end = ptr[row + 1];
        for (int i = row_start; i < row_end; i++)
        {
            sum_of_row += val[i] * vec[col[i]];
        }
        Ap[row] = sum_of_row;
    }
}

double getResidual_csr()
{
    double residual = 0;
    for (int row = 0; row < num_rows; row++)
    {
        //double sum_of_row = 0;
        // int row_start = ptr[row];
        // int row_end = ptr[row + 1];
        // for (int i = row_start; i < row_end; i++)
        // {
        // 	sum_of_row += val[i] * x[col[i]];
        // }
        // residual += (b[row] - sum_of_row) * (b[row] - sum_of_row);
        residual += r1[row] * r1[row];
    }
    return sqrt(residual);
}

double multiVector(double *a, double *b)
{
    double result = 0;
    for (int i = 0; i < num_rows; i++)
        result += a[i] * b[i];
    return result;
}

void replaceVector(double *a, double *b)
{
    for (int i = 0; i < num_rows; i++)
        a[i] = b[i];
}
