#include "error.cuh"
#include <cmath>
#include <cstdio>

__device__ __managed__ int ret[1000];

__global__ void AplusB(int a, int b)
{
    ret[threadIdx.x] = a + b + threadIdx.x;
}

int main(int argc, char *argv[])
{
    AplusB<<<1, 1000>>>(10, 100);
    cudaDeviceSynchronize();
    for (int i = 0; i < 1000; i++)
    {
        printf("%d: A+B = %d\n", i, ret[i]);
    }
}