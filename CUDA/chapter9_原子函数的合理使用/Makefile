all: reduce neighbor1cpu neighbor2gpu

reduce: reduce.cu
	nvcc -g -arch=sm_50 reduce.cu -o reduce

neighbor1cpu: neighbor1cpu.cu
	nvcc -g -arch=sm_50 neighbor1cpu.cu -o neighbor1cpu

neighbor2gpu: neighbor2gpu.cu
	nvcc -g -arch=sm_50 neighbor2gpu.cu -o neighbor2gpu

.PHONY: clean

clean:
	rm -rf reduce neighbor1cpu neighbor2gpu
