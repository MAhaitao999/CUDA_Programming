all: matrix

matrix: matrix.cu
	nvcc -arch=sm_50 matrix.cu -o matrix
.PHONY: clean

clean:
	rm -rf matrix
