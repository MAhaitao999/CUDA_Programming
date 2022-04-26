all: simpleHyperqOpenmp simpleMultiAddBreadth simpleMultiAddDepth host-kernel kernel-kernel kernel-transfer pinMemTransfer

simpleHyperqOpenmp: simpleHyperqOpenmp.cu
	nvcc -O3 -Xcompiler -fopenmp simpleHyperqOpenmp.cu -o simpleHyperqOpenmp -lgomp

simpleMultiAddBreadth: simpleMultiAddBreadth.cu
	nvcc -O3 -Xcompiler -fopenmp simpleMultiAddBreadth.cu -o simpleMultiAddBreadth -lgomp

simpleMultiAddDepth: simpleMultiAddDepth.cu
	nvcc -O3 -Xcompiler -fopenmp simpleMultiAddDepth.cu -o simpleMultiAddDepth -lgomp

host-kernel: host-kernel.cu
	nvcc -O3 -Xcompiler -fopenmp host-kernel.cu -o host-kernel -lgomp

kernel-kernel: kernel-kernel.cu
	nvcc -O3 -Xcompiler -fopenmp kernel-kernel.cu -o kernel-kernel -lgomp

kernel-transfer: kernel-transfer.cu
	nvcc -O3 -Xcompiler -fopenmp kernel-transfer.cu -o kernel-transfer -lgomp

pinMemTransfer: pinMemTransfer.cu
	nvcc -O3 pinMemTransfer.cu -o pinMemTransfer

.PHONY: clean

clean:
	rm -rf simpleHyperqOpenmp simpleMultiAddBreadth simpleMultiAddDepth host-kernel kernel-kernel kernel-transfer pinMemTransfer
