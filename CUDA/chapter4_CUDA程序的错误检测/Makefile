all: check1api check2kernel memcheck

check1api: check1api.cu
	nvcc check1api.cu -o check1api

check2kernel: check2kernel.cu
	nvcc check2kernel.cu -o check2kernel

memcheck: memcheck.cu
	nvcc memcheck.cu -o memcheck

.PHONY: clean

clean:
	rm -rf check1api check2kernel memcheck
