#include <cstdio>
#include <cstdlib>
#include <curand.h>

void output_results(int N, double *g_x);

int main(int argc, char *argv[])
{
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 1234);
    int N = 100000;
    double *g_x;
    cudaMalloc((void **)&g_x, sizeof(double) * N);
    curandGenerateNormalDouble(generator, g_x, N, 0.0, 1.0);
    double *x = (double *)calloc(N, sizeof(double));
    cudaMemcpy(x, g_x, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(g_x);
    output_results(N, x);
    free(x);

    return 0;
}

void output_results(int N, double *x)
{
    FILE *fid = fopen("x2.txt", "w");
    for(int n = 0; n < N; n++)
    {
        fprintf(fid, "%g\n", x[n]);
    }
    fclose(fid);
}