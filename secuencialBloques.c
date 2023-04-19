#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 25 // Tamaño de la matriz
#define M 25 // Tamaño del bloque

int main() {
    int i, j, k, i1, j1, k1;
    double sum;
    double **A, **B, **C;
    struct timespec begin, end;

    // Reserva de memoria para las matrices
    A = (double **) calloc(N , sizeof(double *));
    B = (double **) calloc(N , sizeof(double *));
    C = (double **) calloc(N , sizeof(double *));
    for (i = 0; i < N; i++) {
        A[i] = (double *) calloc(N , sizeof(double));
        B[i] = (double *) calloc(N , sizeof(double));
        C[i] = (double *) calloc(N , sizeof(double));
    }

    // Inicialización de las matrices A y B con valores aleatorios
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (double) rand()*10.0 / RAND_MAX; // valores aleatorios entre 0 y 10
            B[i][j] = (double) rand()*10.0 / RAND_MAX; // valores aleatorios entre 0 y 10
        }
    }

    // Multiplicación de matrices
    clock_gettime(CLOCK_REALTIME, &begin);
    // Recorrer los bloques de las matrices
    for (i = 0; i < N; i += M) {
        for (j = 0; j < N; j += M) {
            for (k = 0; k < N; k += M) {
                // Multiplicar los bloques correspondientes
                for (i1 = i; i1 < i+M; i1++) {
                    for (j1 = j; j1 < j+M; j1++) {
                        for (k1 = k; k1 < k+M; k1++) {
                            C[i1][j1] += A[i1][k1] * B[k1][j1];
                        }
                    }
                }
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);

    // Impresión de los resultados
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;
    printf("The elapsed time is %8.7f seconds", elapsed);
    printf("\n");

    printf("Matriz A:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%.2f ", A[i][j]);
        }   
        printf("\n");
    }

    printf("Matriz B:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%.2f ", B[i][j]);
        }   
        printf("\n");
    }

    printf("Matriz C:\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%.2f ", C[i][j]);
        }   
        printf("\n");
    }

    // Liberación de memoria
    for (i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}

