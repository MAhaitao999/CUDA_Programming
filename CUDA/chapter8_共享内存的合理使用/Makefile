all: reduce1cpu reduce2gpu bank

reduce1cpu: reduce1cpu.cu
	nvcc -arch=sm_50 reduce1cpu.cu -o reduce1cpu

reduce2gpu: reduce2gpu.cu
	nvcc -arch=sm_50 reduce2gpu.cu -o reduce2gpu

bank: bank.cu
	nvcc -arch=sm_50 bank.cu -o bank

.PHONY: clean

clean:
	rm -rf reduce1cpu reduce2gpu bank
