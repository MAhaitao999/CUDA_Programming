all: hello1 hello2 hello3 hello4 hello5

hello1: hello1.cpp
	g++ hello1.cpp -o hello1

hello2: hello2.cu
	nvcc hello2.cu -o hello2

hello3: hello3.cu
	nvcc hello3.cu -o hello3

hello4: hello4.cu
	nvcc hello4.cu -o hello4

hello5: hello5.cu
	nvcc hello5.cu -o hello5

.PHONY: clean

clean:
	rm -rf hello1 hello2 hello3 hello4 hello5
