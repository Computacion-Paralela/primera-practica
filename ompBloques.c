// gcc ompBloques.c -o ompBloques -fopenmp
// ./ompBloques

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Definición del tamaño del bloque
#define M 8

// Función para guardar la matriz en un archivo
void guardarMatriz(double **matriz, int nFilas, int nColumnas, char *nombreArchivo) {
    FILE *archivo;
    archivo = fopen(nombreArchivo, "w");

    if (archivo == NULL) {
        printf("Error al abrir el archivo.\n");
        return;
    }

    for (int i = 0; i < nFilas; i++) {
        for (int j = 0; j < nColumnas; j++) {
            fprintf(archivo, "%lf ", matriz[i][j]);
        }
        fprintf(archivo, "\n");
    }

    fclose(archivo);
}

int main() {

    // Definición del tamaño de la matriz
    int N;
    printf("Ingrese el tamaño de la matriz (N): ");
    scanf("%d", &N);

    //Definición del númerp de hilos
    int NUM_HILOS;
    printf("Ingrese el número de hilos (NUM_HILOS): ");
    scanf("%d", &NUM_HILOS);

    // Inicialización de las variables
    int i, j, k, i1, j1, k1;
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
                // private(i,j,k,sum) indica que las variables j, k y sum son privadas para cada hilo, lo que significa que cada hilo tiene su propia copia de estas variables y no pueden ser compartidas entre hilos.
                // shared(A,B,C) indica que las matrices A, B y C se compartirán entre los hilos y se les permitirá acceder a ellas y modificarlas.
                #pragma omp parallel for private(i, j, k, i1, j1, k1) shared(A,B,C) num_threads(NUM_HILOS)
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

    // Calcular el tiempo transcurrido
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;
    printf("The elapsed time is %8.7f seconds", elapsed);
    printf("\n");

    // Impresión de los resultados
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

    // Guardado de las matrices en sus respectivos archivos
    char *matrizA = "matrizA.txt";
    guardarMatriz(A, N, N, matrizA);

    char *matrizB = "matrizB.txt";
    guardarMatriz(B, N, N, matrizB);

    char *matrizC = "matrizC.txt";
    guardarMatriz(C, N, N, matrizC);

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