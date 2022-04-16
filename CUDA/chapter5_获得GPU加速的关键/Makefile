all: add1cpu add2gpu add3memcpy arithmetic1cpu arithmetic2gpu

add1cpu: add1cpu.cu
	nvcc -O3 -DUSE_DP add1cpu.cu -o add1cpu

add2gpu: add2gpu.cu
	nvcc -O3 -DUSE_DP add2gpu.cu -o add2gpu

add3memcpy: add3memcpy.cu
	nvcc -O3 -DUSE_DP add3memcpy.cu -o add3memcpy

arithmetic1cpu: arithmetic1cpu.cu
	nvcc -O3 -DUSE_DP arithmetic1cpu.cu -o arithmetic1cpu

arithmetic2gpu: arithmetic2gpu.cu
	nvcc -O3 -DUSE_DP arithmetic2gpu.cu -o arithmetic2gpu

.PHONY: clean

clean:
	rm -rf add1cpu add2gpu add3memcpy arithmetic1cpu arithmetic2gpu
