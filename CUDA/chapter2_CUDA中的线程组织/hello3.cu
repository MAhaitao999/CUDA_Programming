#include <cstdio>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(int argc, char *argv[])
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}