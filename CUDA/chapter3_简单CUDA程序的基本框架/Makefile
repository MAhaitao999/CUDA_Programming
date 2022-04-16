all: add add1 add2wrong add3if add4device

add: add.cpp
	g++ add.cpp -o add

add1: add1.cu
	nvcc add1.cu -o add1

add2wrong: add2wrong.cu
	nvcc add2wrong.cu -o add2wrong

add3if: add3if.cu
	nvcc add3if.cu -o add3if

add4device: add4device.cu
	nvcc add4device.cu -o add4device

.PHONY: clean

clean:
	rm -rf add add1 add2wrong add3if add4device
