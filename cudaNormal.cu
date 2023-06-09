//%%cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCKS_PER_GRID 16
#define THREADS_PER_BLOCK 16

void inicializar_matriz(int n, float *matriz) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = rand() % 10;
    }
}

void imprimir_matriz(int n, float *matriz) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matriz[i * n + j]);
        }
        printf("\n");
    }
}

void guardar_matriz_txt(int n, float *matriz, const char *nombre_archivo) {
    FILE *archivo = fopen(nombre_archivo, "w");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(archivo, "%.2f ", matriz[i * n + j]);
        }
        fprintf(archivo, "\n");
    }
    fclose(archivo);
}

__global__ void matriz_multiplicar(float *a, float *b, float *c, int n) {
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float suma = 0;
    if (fila < n && col < n) {
        for (int i = 0; i < n; i++) {
            suma += a[fila * n + i] * b[i * n + col];
        }
        c[fila * n + col] = suma;
    }
}

int main() {

    int n;
    printf("Ingrese el tamaño de la matriz (n): ");
    scanf("%d", &n);

    struct timespec begin, end;
    
    // Asignar memoria en el host
    float *matriz_a = (float *) malloc(sizeof(float) * n * n);
    float *matriz_b = (float *) malloc(sizeof(float) * n * n);
    float *matriz_res = (float *) malloc(sizeof(float) * n * n);

    // Inicializar matrices
    inicializar_matriz(n, matriz_a);
    inicializar_matriz(n, matriz_b);

    // Imprimir matrices originales
    //printf("Matriz A:\n");
    //imprimir_matriz(n, matriz_a);
    //printf("Matriz B:\n");
    //imprimir_matriz(n, matriz_b);

    // Asignar memoria en el dispositivo
    float *d_a, *d_b, *d_res;
    cudaMalloc((void **) &d_a, sizeof(float) * n * n);
    cudaMalloc((void **) &d_b, sizeof(float) * n * n);
    cudaMalloc((void **) &d_res, sizeof(float) * n * n);

    // Copiar matrices del host al dispositivo
    cudaMemcpy(d_a, matriz_a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, matriz_b, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    // Definir la cantidad de threads 
    dim3 block_size(BLOCKS_PER_GRID, BLOCKS_PER_GRID, 1);
    dim3 grid_size(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    // Lanzar kernel para multiplicar matrices 
    clock_gettime(CLOCK_REALTIME, &begin);

    matriz_multiplicar<<<grid_size, block_size>>>(d_a, d_b, d_res, n);

    clock_gettime(CLOCK_REALTIME, &end);

    // Calcular el tiempo transcurrido
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;

    printf("The elapsed time is %8.7f seconds", elapsed);
    printf("\n");

    // Copiar matriz resultado del dispositivo al host
    cudaMemcpy(matriz_res, d_res, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    // Imprimir matriz resultado
    //printf("Matriz resultado:\n");
    //imprimir_matriz(n, matriz_res);
    guardar_matriz_txt(n,matriz_a, "matrizA.txt");
    guardar_matriz_txt(n,matriz_b, "matrizB.txt");
    guardar_matriz_txt(n,matriz_res, "matrizC.txt");

    // Liberar memoria
    free(matriz_a);
    free(matriz_b);
    free(matriz_res);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return 0;
}