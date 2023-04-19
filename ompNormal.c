// gcc ompNormal.c -o ompNormal -fopenmp
// ./ompNormal

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 1024 // Tamaño de la matriz
#define NUM_HILOS 1 // Número de hilos

struct timespec begin, end;

int main(){

    int i, j, k;
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

    // private(i,j,k,sum) indica que las variables j, k y sum son privadas para cada hilo, lo que significa que cada hilo tiene su propia copia de estas variables y no pueden ser compartidas entre hilos.
    // shared(A,B,C) indica que las matrices A, B y C se compartirán entre los hilos y se les permitirá acceder a ellas y modificarlas.
    #pragma omp parallel for private(i,j,k,sum) shared(A,B,C) num_threads(NUM_HILOS)
    // debido al uso de pragma omp parallel for no se colocan los corchetes de barrera
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.0;
            for (k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    clock_gettime(CLOCK_REALTIME, &end);


    // Calcular el tiempo transcurrido
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;

    printf("The elapsed time is %8.7f seconds", elapsed);
    printf("\n");

    // printf("Matriz A:\n");
    // for (i = 0; i < N; i++) {
    //     for (j = 0; j < N; j++) {
    //         printf("%.2f ", A[i][j]);
    //     }   
    //     printf("\n");
    // }

    // printf("Matriz B:\n");
    // for (i = 0; i < N; i++) {
    //     for (j = 0; j < N; j++) {
    //         printf("%.2f ", B[i][j]);
    //     }   
    //     printf("\n");
    // }

    // printf("Matriz C:\n");
    // for (i = 0; i < N; i++) {
    //     for (j = 0; j < N; j++) {
    //         printf("%.2f ", C[i][j]);
    //     }   
    //     printf("\n");
    // }

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