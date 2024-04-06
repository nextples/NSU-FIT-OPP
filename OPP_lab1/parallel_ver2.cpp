#include <cstdio>
#include <openmpi/mpi.h>
#include <math.h>
#include <cassert>

double** createMatrix (int n, int begin, int len);

void deleteMatrix (double** matrix, int m);

double* createSinVector (int fullN, int m, int maxM, double t, int begin);

double* createVector (double val, int maxM);

void deleteVector (double* vector);

double* multMatrixVector (double **matrix, double *vec, int maxM, int fullN, int* begins, int* lens, int rank, int size);

void sub (double* mainVector, double* vector, int n);

void multByScalar (double* mainVector, double scalar, int n);

double updateXVector (double** matrix, double* bVector, double* xVector, int nn, int fullN, int* begin, int* lens, int rank, int size);

double scalarMult (double* vector1, double* vector2, int n);

void printVector (const double* vector, int n);

int* createBegin (int fullN, int size);

int* createLen (int *begin, int fullN, int size);

int main (int argc, char **argv) {
    int size = 0, rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int fullN = 10000;
    double pi = 3.1415;
    int* begins = createBegin(fullN, size);
    int* lens = createLen(begins, fullN, size);
    int m = lens[rank];
    int maxM = lens[0];

    double** matrix = createMatrix(fullN, begins[rank], lens[rank]);
    double* uVector = createSinVector(fullN, m, maxM, pi, begins[rank]);
//    printf("check1\n");
    double* bVector = multMatrixVector(matrix, uVector, maxM, fullN, begins, lens, rank, size);
//    printf("check2\n");
    double* xVector = createVector(0, maxM);
//    printf("check3\n");

    double expectedAnswer = scalarMult(bVector, bVector, lens[rank]);
    if (rank == 0) {
        printf("expected answer - //%f//\n", expectedAnswer);
    }

    double e = 1;
    double eps = 0.00001;
    double start, end;

    start = MPI_Wtime();
    while (e > eps) {
        e = updateXVector(matrix, bVector, xVector, maxM, fullN, begins, lens, rank, size);
    }
    end = MPI_Wtime();

    double gotAnswer = scalarMult(xVector, xVector, lens[rank]);
    if (rank == 0) {
        printf("got answer - %lf\n", gotAnswer);
        double time = end - start;
        printf("time: %lf\n", time);
    }

    deleteMatrix(matrix, lens[rank]);
    deleteVector(bVector);
    deleteVector(uVector);
    deleteVector(xVector);

    free(begins);
    free(lens);
    MPI_Finalize();
    return 0;
}

double** createMatrix(int n, int begin, int len) {
    double **matrix;
    matrix = (double**) malloc(len * sizeof(double*));
    assert(matrix != NULL);
    for (int i = 0; i < len; i++) {
        matrix[i] = (double*) malloc(n * sizeof(double));
        assert(matrix[i] != NULL);
    }
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = 1 + ((begin + i) == j);
        }
    }
    return matrix;
}

void deleteMatrix(double** matrix, const int m) {
    for (int i = 0; i < m; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

double* createSinVector(int fullN, int m, int maxM, double t, int begin) {
    double* vector = (double*) malloc(maxM * sizeof(double));
    for (int i = begin; i < begin + m; i++) {
        vector[i - begin] = sin((2 * t * i) / fullN);
    }
    return vector;
}

double* createVector(double val, int maxM) {
    double* vector = (double*) malloc(maxM * sizeof(double));
    for (int i = 0; i < maxM; i++) {
        vector[i] = val;
    }
    return vector;
}

void deleteVector(double* vector) {
    free(vector);
}

double* multMatrixVector(double **matrix, double *vec, int maxM, int fullN, int* begins, int* lens, int rank, int size) {
    int strNumb = lens[rank];
    double* result =(double*) calloc(maxM, sizeof(double));

    for (int k = 0; k < size; k++) {
        int begin = begins[(rank + size - k) % size];
        int localN = lens[(rank + size - k) % size];

        for (int i = 0; i < strNumb; i++) {
            int jFull = begin;
            for (int j = 0; j < localN; j++) {
                result[i] += (matrix[i][jFull] * vec[j]);
                jFull++;
            }
        }
        MPI_Sendrecv_replace(vec,
                             maxM,
                             MPI_DOUBLE,
                             (rank + 1) % size,
                             123,
                             (rank + size - 1) % size,
                             123,
                             MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE
        );
    }
    return result;
}

void sub(double* mainVector, double* vector, int n) {
    for (int i = 0; i < n; i++) {
        mainVector[i] -= vector[i];
    }
}

void multByScalar(double* mainVector, double scalar, int n) {
    for (int i = 0; i < n; i++) {
        mainVector[i] *= scalar;
    }
}

double scalarMult(double *vector1, double *vector2, int n) {
    double mult = 0;
    for (int i = 0; i < n; i++) {
        mult += vector1[i] * vector2[i];
    }
    double result = 0;
    MPI_Allreduce(&mult, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return result;
}

double updateXVector(double** matrix, double* bVector, double* xVector, int nn, int fullN, int* begin, int* lens, int rank, int size) {
    int m = lens[rank];
    double* yVector = multMatrixVector(matrix, xVector, nn, fullN, begin, lens, rank, size);

    sub(yVector, bVector, m);
    double e = sqrt(scalarMult(yVector, yVector, m));
    double b = sqrt(scalarMult(bVector, bVector, m));

    double* AyVector = multMatrixVector(matrix, yVector, nn, fullN, begin, lens, rank, size);
    double tau = scalarMult(yVector, AyVector, m) / scalarMult(AyVector, AyVector, m);
    multByScalar(yVector, tau, m);
    sub(xVector, yVector, m);

    deleteVector(yVector);
    deleteVector(AyVector);

    return e / b;
}

void printVector(const double* vector, const int n) {
    printf("vector: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}

int* createBegin(int fullN, int size) {
    int* begin = (int*) malloc(size * sizeof(int));
    begin[0] = 0;
    for (int i = 1; i < size; i++) {
        begin[i] = begin[i - 1] + (fullN / size) + ((fullN % size) >= i);
    }
    return begin;
}

int* createLen(int *begin, int fullN, int size) {
    int* len = (int*) malloc(size * sizeof(int));
    for (int i = 0; i < size - 1; i++) {
        len[i] = begin[i + 1] - begin[i];
    }
    len[size - 1] = fullN - begin[size - 1];
    return len;
}