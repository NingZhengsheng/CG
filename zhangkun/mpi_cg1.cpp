#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <iomanip>
#include <omp.h>
#include <mpi.h>

using namespace std;
#define MAX_ITERATION 200
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

//dot temp  分子分母
double numerator, denominator;
double dot_part;

double threshold = THRESHOLD;
int iteration = 0;
double norm = 0;

double *r0;
double *r1;
double *z0;
double *z1;
double *y;
double beta;
double alpha;
double *p;
double *Ap;
//subvector
double *r1_sub;
double *p_sub;

string field = "";
string symmetry = "";

string filename = "Dubcova2";
int NUM_OF_THREADS; //omp threads
int worldRank, worldSize, rowPart, lastPart;
double spmvtime = 0, setpretime = 0, usepretime = 0, dottime = 0, axpytime = 0;
double spmvts, spmvte, setpres, setpree, usepres, usepree, dotts, dotte, axpyts, axpyte;

int main(int argc, char **argv)
{
	if (argc > 1)
		filename = argv[1];
	MPI_Init(NULL, NULL);					   //mpi init
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize); //process num
	MPI_Comm_rank(MPI_COMM_WORLD, &worldRank); //process rank
	MPI_Request request1;
	MPI_Status status1;

	int namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Get_processor_name(processor_name, &namelen);
	printf(" process id = %d ,process total =%d , host = %s\n", worldRank, worldSize, processor_name);
	MPI_Barrier(MPI_COMM_WORLD);
	void readMatrix();
	void csr();											 //change format  coo2csr
	void ilu0();										 //init ilu0 matrix
	void LU(double *r, double *z);						 //sptrsv
	void Jacobi(double *r, double *z);					 //z=(p^-1)r
	void vectorSynch(double *send, double *receive);	 //synchronize vector
	void initData();									 //p=r1=r0=b-Ax
	void getResidual();									 //norm=||r1||2
	void SpMV_A(double *vec);							 //spmv  A
	void multiVector(double *a, double *b);				 //dot
	void SMV(double *a, double *b, double c, double *d); //scalar multiply vector a=b+c*d
	void replaceVector(double *a, double *b);			 //copy vector a=b

	readMatrix();
	csr();
	setpres = MPI_Wtime();
	ilu0();
	setpree = MPI_Wtime();
	setpretime = setpree - setpres;
	rowPart = num_rows / worldSize;	 //num of rows calculated by each process
	lastPart = num_rows % worldSize; //the last process calculates more elements

	double startTime, endTime;
	startTime = MPI_Wtime();
	spmvts = MPI_Wtime();
	initData(); //precondition
	spmvte = MPI_Wtime();
	spmvtime += spmvte - spmvts;
	// initData();
	MPI_Allgather(r1_sub + worldRank * rowPart, rowPart, MPI_DOUBLE, r1, rowPart, MPI_DOUBLE, MPI_COMM_WORLD);
	vectorSynch(r1_sub, r1); //send r1_sub to all r1
	usepres = MPI_Wtime();
	//LU(r1, z0);
	Jacobi(r1,z0);
	usepree = MPI_Wtime();
	usepretime += usepree - usepres;
	axpyts = MPI_Wtime();
	for (int i = 0; i < num_rows; i++)
	{
		p[i] = z0[i];
		p_sub[i] = z0[i];
	}
	axpyte = MPI_Wtime();
	axpytime += axpyte - axpyts;
	// MPI_Allgather(p_sub + worldRank * rowPart, rowPart, MPI_DOUBLE, p, rowPart, MPI_DOUBLE, MPI_COMM_WORLD);
	// vectorSynch(p_sub, p); //send p_sub to all p
	dotts = MPI_Wtime();
	getResidual();
	dotte = MPI_Wtime();
	dottime += dotte - dotts;
// while (iteration < MAX_ITERATION && norm > threshold) //02
	while (iteration < MAX_ITERATION ) //02
	{

		spmvts = MPI_Wtime();
		SpMV_A(p);
		spmvte = MPI_Wtime();
		spmvtime += spmvte - spmvts;

		dotts = MPI_Wtime();
		multiVector(r0, z0); //precondition
		// multiVector(r0, r0);
		MPI_Iallreduce(&dot_part, &numerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request1);
		multiVector(Ap, p);
		MPI_Allreduce(&dot_part, &denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Wait(&request1, &status1);
		alpha = numerator / denominator;
		dotte = MPI_Wtime();
		dottime += dotte - dotts;

		axpyts = MPI_Wtime();
		SMV(x, x, alpha, p);
		SMV(r1_sub, r0, -alpha, Ap); //precondition
		axpyte = MPI_Wtime();
		axpytime += axpyte - axpyts;
		// SMV(r1, r0, -alpha, Ap);

		MPI_Allgather(r1_sub + worldRank * rowPart, rowPart, MPI_DOUBLE, r1, rowPart, MPI_DOUBLE, MPI_COMM_WORLD);
		vectorSynch(r1_sub, r1); //send r1_sub to all r1
		usepres = MPI_Wtime();
		//LU(r1, z1);
		Jacobi(r1,z1);
		usepree = MPI_Wtime();
		usepretime += usepree - usepres;

		dotts = MPI_Wtime();
		multiVector(r1, z1); //precondition
		// multiVector(r1, r1);
		MPI_Iallreduce(&dot_part, &numerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request1);
		multiVector(r0, z0); //precondition
		// multiVector(r0, r0);
		MPI_Allreduce(&dot_part, &denominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Wait(&request1, &status1);
		beta = numerator / denominator;
		dotte = MPI_Wtime();
		dottime += dotte - dotts;

		axpyts = MPI_Wtime();
		SMV(p_sub, z1, beta, p); //precondition
		// SMV(p_sub, r1, beta, p);
		axpyte = MPI_Wtime();
		axpytime += axpyte - axpyts;

		MPI_Iallgather(p_sub + worldRank * rowPart, rowPart, MPI_DOUBLE, p, rowPart, MPI_DOUBLE, MPI_COMM_WORLD, &request1);
		vectorSynch(p_sub, p); //send p_sub to all p

		axpyts = MPI_Wtime();
		replaceVector(r0, r1);
		replaceVector(z0, z1); //precondition
		axpyte = MPI_Wtime();
		axpytime += axpyte - axpyts;
		iteration++;
		dotts = MPI_Wtime();
		getResidual();
		dotte = MPI_Wtime();
		dottime += dotte - dotts;
		MPI_Wait(&request1, &status1);
		if (worldRank == 0)
		{
			cout << "ite:" << iteration;
			cout << "  res:" << scientific << setprecision(3) << norm << endl;
		}
	}
	endTime = MPI_Wtime();
	getResidual();
	if (worldRank == 0)
	{
		cout << "***********************" << endl;
		cout << "num of iteration:" << iteration << endl;
		cout << "residual:" << norm << endl;
		cout << "setpretime" << fixed << setprecision(3) << setpretime << endl;
		cout << "***********************" << endl;
		cout << "runTime:" << fixed << setprecision(3) << endTime - startTime << "s" << endl;
		cout << "spmvtime" << fixed << setprecision(3) << spmvtime << endl;
		cout << "usepretime" << fixed << setprecision(3) << usepretime << endl;
		cout << "dottime" << fixed << setprecision(3) << dottime << endl;
		cout << "axpytime" << fixed << setprecision(3) << axpytime << endl;
	}
	MPI_Finalize();
	return 0;
}
/**
 *read matrix file  coo
 */
void readMatrix()
{
	//cout << "input matrix filename!" << endl;
	//cin>>filename;
	//filename = "t";
	ifstream fin;
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
		if (worldRank == 0)
		{
			cout << "num_rows:" << num_rows << "  num_cols:" << num_col << "  num_elements(symmetric):" << num_elements << endl;
		}
	}
	else
	{
		if (worldRank == 0)
		{
			cout << "num_rows:" << num_rows << "  num_cols:" << num_col << "  num_elements(general):" << num_elements << endl;
		}
	}
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

	// CSR
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
	//subvector needs synchronization
	r1_sub = new double[num_rows];
	p_sub = new double[num_rows];
	memset(ptr_cnt, 0, num_rows * sizeof(int)); //set 0
}

/**
 *coo2csr
 */
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
	int offset;									//start of each row
	for (int i = 0; i < num_elements; i++)
	{
		offset = ptr[row[i]] + ptr_cnt[row[i]];
		col[offset] = column[i];
		val[offset] = value[i];
		ptr_cnt[row[i]]++;
	}

	// int flag_col = 0;
	// for (int i = 0; i < num_rows; i++)
	// {
	// 	for (int j = 0; j < num_elements; j++)
	// 	{
	// 		if (row[j] == i)
	// 		{
	// 			col[flag_col] = column[j];
	// 			val[flag_col] = value[j];
	// 			flag_col++;
	// 		}
	// 	}
	// 	ptr[i + 1] = flag_col;
	// }

	//sort
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

			startk = ptr[col[k]];	//col k 行 start
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

/**
 * synchronize vector
 */
void vectorSynch(double *send, double *receive)
{

	int row;
	MPI_Bcast(send + worldSize * rowPart, lastPart, MPI_DOUBLE, worldSize - 1, MPI_COMM_WORLD);
	for (row = worldSize * rowPart; row < worldSize * rowPart + lastPart; row++)
	{
		receive[row] = send[row];
	}
}
/**
 *init data, p=r1=r0=b-Ax,
 */
void initData()
{
	double sum_of_row;
	for (int row = worldRank * rowPart; row < (worldRank + 1) * rowPart; row++)
	{
		sum_of_row = 0;
		int row_start = ptr[row];
		int row_end = ptr[row + 1];
		for (int i = row_start; i < row_end; i++)
		{
			sum_of_row += val[i] * x[col[i]];
		}
		r0[row] = (b[row] - sum_of_row);
		r1_sub[row] = r0[row];
		p_sub[row] = r0[row];
	}

	if (lastPart != 0 && worldRank == (worldSize - 1))
	{
		for (int row = worldSize * rowPart; row < worldSize * rowPart + lastPart; row++)
		{
			sum_of_row = 0;
			int row_start = ptr[row];
			int row_end = ptr[row + 1];
			for (int i = row_start; i < row_end; i++)
			{
				sum_of_row += val[i] * x[col[i]];
			}
			r0[row] = (b[row] - sum_of_row);
			r1_sub[row] = r0[row];
			p_sub[row] = r0[row];
		}
	}
}

/**
 *norm=||r1||2
 */
void getResidual()
{
	double residual = 0;
	for (int row = worldRank * rowPart; row < (worldRank + 1) * rowPart; row++)
	{
		residual += r1[row] * r1[row];
	}
	if (lastPart != 0 && worldRank == (worldSize - 1))
	{
		for (int row = worldSize * rowPart; row < worldSize * rowPart + lastPart; row++)
		{
			residual += r1[row] * r1[row];
		}
	}
	MPI_Allreduce(&residual, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	norm = sqrt(norm);
}

/**
 * spmv  A
 */
void SpMV_A(double *vec)
{
	double sum_of_row;
	for (int row = worldRank * rowPart; row < (worldRank + 1) * rowPart; row++)
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

	if (lastPart != 0 && worldRank == (worldSize - 1))
	{
		for (int row = worldSize * rowPart; row < worldSize * rowPart + lastPart; row++)
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
}

/**
 * dot
 */
void multiVector(double *a, double *b)
{
	// double result = 0;
	// for (int i = 0; i < num_rows; i++)
	// 	result += a[i] * b[i];
	// return result;
	dot_part = 0;

	for (int row = worldRank * rowPart; row < (worldRank + 1) * rowPart; row++)
	{
		dot_part += a[row] * b[row];
	}
	if (lastPart != 0 && worldRank == (worldSize - 1))
	{
		for (int row = worldSize * rowPart; row < worldSize * rowPart + lastPart; row++)
		{
			dot_part += a[row] * b[row];
		}
	}
}
/**
 *scalar multiply vector a=b+c*d
 */
void SMV(double *a, double *b, double c, double *d)
{
	for (int row = worldRank * rowPart; row < (worldRank + 1) * rowPart; row++)
	{
		a[row] = b[row] + c * d[row];
	}
	if (lastPart != 0 && worldRank == (worldSize - 1))
	{
		for (int row = worldSize * rowPart; row < worldSize * rowPart + lastPart; row++)
		{
			a[row] = b[row] + c * d[row];
		}
	}
}
/**
 * copy vector  a=b
 */
void replaceVector(double *a, double *b)
{
	for (int row = worldRank * rowPart; row < (worldRank + 1) * rowPart; row++)
	{
		a[row] = b[row];
	}
	if (lastPart != 0 && worldRank == (worldSize - 1))
	{
		for (int row = worldSize * rowPart; row < worldSize * rowPart + lastPart; row++)
		{
			a[row] = b[row];
		}
	}
}
