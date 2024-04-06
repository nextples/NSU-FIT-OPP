#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

double** createMatrix (const int n);

void deleteMatrix (double** matrix, const int m);

double* createSinVector (const int n, const double t);

double* createVector (const double val, const int maxM);

void deleteVector (double* vector);

void mulMatrixVector (double **matrix, double *vector, double *res, const int n);

void sub (double* mainVector, const double* vector, const int n);

void mulByScalar (double* vector, double scalar, int n);

double updateX (double** matrix, double* bVector, double* xVector, double* yVector, double* AyVector, const int n);

void setVectorZero (double* vector, int n);

double scalarMul (const double* vector1, const double* vector2, const int n);

void printVector (const double* vector, const int n);


int main () {
    int n = 10000;
    double pi = 3.1415;

    double** matrix = createMatrix(n);
    double* uVector = createSinVector(n, pi);
    double* bVector = createVector(0, n);
    mulMatrixVector(matrix, uVector, bVector, n);
    double* xVector = createVector(0, n);
    double* yVector = createVector(0, n);
    double* AyVector = createVector(0, n);

    printf("expected answer - //%f//\n", scalarMul(bVector, bVector, n));
    double e = 1;
    double eps = 0.00001;
    int cnt = 0;
    time_t start, end;

    start = clock();
    while (e > eps) {
        cnt++;
        e = updateX(matrix, bVector, xVector, yVector, AyVector, n);
    }
    end = clock();

    double time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("time: %f\n", time);
    printf("got answer - %lf\n", scalarMul(xVector, xVector, n));

    deleteMatrix(matrix, n);
    deleteVector(bVector);
    deleteVector(uVector);
    deleteVector(xVector);
    deleteVector(AyVector);
    deleteVector(yVector);
}

double** createMatrix (const int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                matrix[i][j] = 2.0;
            }
            else {
                matrix[i][j] = 1.0;
            }
        }
    }
    return matrix;
}

void deleteMatrix (double** matrix, const int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

double* createSinVector (const int n, const double t) {
    double* vector = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        vector[i] = sin((2 * t * i) / n);
    }
    return vector;
}

double* createZeroVector (const int n) {
    double* vector = (double*)calloc(n, sizeof(double));
    return vector;
}

double* createVector (const double t, const int n) {
    double* vector = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        vector[i] = t;
    }
    return vector;
}

void deleteVector (double* vector) {
    free(vector);
}

void mulMatrixVector(double **matrix, double *vector, double *res, const int n) {
    setVectorZero(res, n);

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i] += matrix[i][j] * vector[j];
        }
    }
}

void sub (double* main_vector, const double* vector, const int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        main_vector[i] -= vector[i];
    }
}

void mulByScalar (double* vector, double scalar, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        vector[i] *= scalar;
    }
}

void setVectorZero(double* vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = 0;
    }
}

double updateX (double** matrix, double* bVector, double* xVector, double* yVector, double* AyVector, const int n) {
    mulMatrixVector(matrix, xVector, yVector, n);
    sub(yVector, bVector, n);

    double yNorm = sqrt(scalarMul(yVector, yVector, n));
    double bNorm = sqrt(scalarMul(bVector, bVector, n));

    mulMatrixVector(matrix, yVector, AyVector, n);
    double tau = scalarMul(yVector, AyVector, n) / scalarMul(AyVector, AyVector, n);
    mulByScalar(yVector, tau, n);
    sub(xVector, yVector, n);

    return yNorm / bNorm;
}

double scalarMul (const double* vector1, const double* vector2, const int n) {
    double mult = 0;
#pragma omp parallel for reduction(+:mult)
    for (int i = 0; i < n; i++) {
        mult += (vector1[i] * vector2[i]);
    }
    return mult;
}

void printVector (const double* vector, const int n) {
    printf("vector: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", vector[i]);
    }
    printf("\n");
}