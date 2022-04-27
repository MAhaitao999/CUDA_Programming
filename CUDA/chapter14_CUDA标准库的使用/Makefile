all: thrust_scan_vector thrust_scan_pointer cublas_gemm cusolver curand_host1 curand_host2

thrust_scan_vector: thrust_scan_vector.cu
	nvcc -arch=sm_60 -O3 thrust_scan_vector.cu -o thrust_scan_vector

thrust_scan_pointer: thrust_scan_pointer.cu
	nvcc -arch=sm_60 -O3 thrust_scan_pointer.cu -o thrust_scan_pointer

cublas_gemm: cublas_gemm.cu
	nvcc -arch=sm_60 -lcublas -O3 cublas_gemm.cu -o cublas_gemm

cusolver: cusolver.cu
	nvcc -arch=sm_60 -lcusolver cusolver.cu -o cusolver

curand_host1: curand_host1.cu
	nvcc -arch=sm_60 -lcurand curand_host1.cu -o curand_host1

curand_host2: curand_host2.cu
	nvcc -arch=sm_60 -lcurand curand_host2.cu -o curand_host2

.PHONY: clean

clean:
	rm -rf thrust_scan_vector thrust_scan_pointer cublas_gemm cusolver curand_host1 curand_host2
