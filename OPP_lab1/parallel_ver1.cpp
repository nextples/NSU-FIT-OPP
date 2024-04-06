#include <cstdio>
#include <openmpi/mpi.h>
#include <math.h>
#include <cassert>

double** createMatrix (int n, int begin, int len);

void deleteMatrix (double** matrix, int n);

int countStrInMatrix (int n, int rank, int size);

void printMatrix (double** matrix, int n, int len, int rank , int size);

double* createSinVector (int n, double t);

double* createVector (double val, int n);

void deleteVector (double* vector);

double *multMatrixVector(double **matrix, double *vector1, int n, int begin, int len);

void sub (double* mainVector, const double* vector, int n);

void multByScalar (double* vector, double scalar, int n);

double updateXVector (double** matrix, double* bVector, double* xVector, int n, int begin, int len);

double scalarMult (const double* vector1, const double* vector2, int n);

void printVector (const double* vector, int n);

int* createBegin (int full_n, int size);

int* createEnd (int *beg, int n, int size);

int main (int argc, char **argv) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = 10000;
    double pi = 3.1415;
    double start, end;

    int* chunkBegin = createBegin(n, size);
    int* chunkEnd = createEnd(chunkBegin, n, size);
    int chunkLen = chunkEnd[rank] - chunkBegin[rank];

    double** matrix = createMatrix(n, chunkBegin[rank], chunkLen);
    double* uVector = createSinVector(n, pi);
    double* bVector = multMatrixVector(matrix, uVector, n, chunkBegin[rank], chunkLen);
    double* xVector = createVector(0, n);

    if (rank == 0) {
        printf("expected answer - //%f//\n", scalarMult(bVector, bVector, n));
    }

    double e = 1;
    double eps = 0.00001;

    start = MPI_Wtime();
    if (rank == 0) {
    }
    
    int cnt = 0;
    while (e > eps) {
        e = updateXVector(matrix, bVector, xVector, n, chunkBegin[rank], chunkLen);
        if (rank == 0) {
            cnt++;
        }
    }
    end = MPI_Wtime();

    if (rank == 0) {
        printf("got answer - %lf\n", scalarMult(xVector, xVector, n));
        double time = end - start;
        printf("time: %lf\n", time);
        printf("cnt = %d\n", cnt);
    }

    deleteMatrix(matrix, chunkLen);
    deleteVector(bVector);
    deleteVector(uVector);
    deleteVector(xVector);

    MPI_Finalize();
    return 0;
}

double** createMatrix (int n, int begin, int len) {
    double **matrix;
    matrix = (double**)malloc(len * sizeof(double*));
    assert(matrix != NULL);
    for (int i = 0; i < len; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
        assert(matrix[i] != NULL);
    }
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = 1 + ((begin + i) == j);
        }
    }

    return matrix;
}

void deleteMatrix (double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void printMatrix (double** matrix, int n, int len, int rank , int size) {
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            printf("rank: %d len: %d\n", rank, len);
            for (int j = 0; j < len; j++) {
                for (int k = 0; k < n; k++) {
                    printf("%lf ", matrix[j][k]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int* createBegin (int full_n, int size) {
    int* begin = (int*) malloc(size * sizeof(int));
    begin[0] = 0;
    for (int i = 1; i < size; i++) {
        begin[i] = begin[i - 1] + (full_n / size) + ((full_n % size) >= i);
    }
    return begin;
}

int* createEnd (int *beg, int n, int size) {
    int* end = (int*) malloc(size * sizeof(int));
    for (int i = 0; i < size - 1; i++) {
        end[i] = beg[i + 1];
    }
    end[size - 1] = n;
    return end;
}

double* createSinVector (int n, double t) {
    double* vector = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        vector[i] = sin((2 * t * i) / n);
    }
    return vector;
}

double* createZeroVector (int n) {
    double* vector = (double*)calloc(n, sizeof(double));
    return vector;
}

double* createVector (double val, int n) {
    double* vector = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        vector[i] = val;
    }
    return vector;
}

void deleteVector (double* vector) {
    free(vector);
}

void setVectorZero(double* vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = 0;
    }
}

double *multMatrixVector(double **matrix, double *vector1, int n, int begin, int len) {
    double *buffVector = createZeroVector(n);
    for (int i = 0; i < len; i++) {
        buffVector[begin + i] = scalarMult(matrix[i], vector1, n);
    }

    double* result = createZeroVector(n);
    MPI_Allreduce(buffVector, result, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    free(buffVector);
    return result;
}

void sub (double* mainVector, const double* vector, int n) {
    for (int i = 0; i < n; i++) {
        mainVector[i] -= vector[i];
    }
}

void multByScalar (double* vector, double scalar, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] *= scalar;
    }
}

double updateXVector (double** matrix, double* bVector, double* xVector, int n, int begin, int len) {
    double* y = multMatrixVector(matrix, xVector, n, begin, len);
    sub(y, bVector, n);

    double e = sqrt(scalarMult(y, y, n));
    double b = sqrt(scalarMult(bVector, bVector, n));

    double* Ay = multMatrixVector(matrix, y, n, begin, len);
    double tau = scalarMult(y, Ay, n) / scalarMult(Ay, Ay, n);
    multByScalar(y, tau, n);
    sub(xVector, y, n);

    deleteVector(y);
    deleteVector(Ay);

    return e / b;
}

double scalarMult (const double* vector1, const double* vector2, int n) {
    double mult = 0;
    for (int i = 0; i < n; i++) {
        mult += (vector1[i] * vector2[i]);
    }
    return mult;
}

void printVector (const double* vector, int n) {
    printf("vector: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}