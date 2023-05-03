//%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

__global__ void matriz_multiplicar_tiling(float *a, float *b, float *c, int n) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * THREADS_PER_BLOCK + ty;
    int col = bx * THREADS_PER_BLOCK + tx;

    float sum = 0;
    for (int i = 0; i < n; i += THREADS_PER_BLOCK) {
        __shared__ float shared_a[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
        __shared__ float shared_b[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
        shared_a[ty][tx] = a[row * n + i + tx];
        shared_b[ty][tx] = b[(i + ty) * n + col];
        __syncthreads();
        for (int j = 0; j < THREADS_PER_BLOCK; j++) {
            sum += shared_a[ty][j] * shared_b[j][tx];
        }
        __syncthreads();
    }
    c[row * n + col] = sum;
}
int main() {
    
    int n;
    printf("Ingrese el tamaño de la matriz (n): ");
    scanf("%d", &n);
    int BLOCKS_PER_GRID;
    printf("Ingrese el número de bloques por cluster (BLOCKS_PER_GRID): ");
    scanf("%d", &n);
    int THREADS_PER_BLOCK;
    printf("Ingrese el número de hilos por bloque (THREADS_PER_BLOCK): ");
    scanf("%d", &n);

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

    // Definir la cantidad de bloques y threads por bloque
    dim3 dimBlock(BLOCKS_PER_GRID, BLOCKS_PER_GRID, 1);
    dim3 dimGrid(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    // Lanzar kernel para multiplicar matrices con tiling
    matriz_multiplicar_tiling<<<dimGrid, dimBlock>>>(d_a, d_b, d_res, n);

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