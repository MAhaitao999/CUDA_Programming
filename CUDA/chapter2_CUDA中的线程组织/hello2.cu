#include <cstdio>
#include <iostream>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
    // std::cout << "Hello world!!" << std::endl;
}

int main(int argc, char *argv[])
{
    hello_from_gpu<<<1, 1>>>();
    // cudaDeviceSynchronize();

    return 0;
}