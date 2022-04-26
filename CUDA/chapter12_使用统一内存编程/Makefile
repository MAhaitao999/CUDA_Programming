all: add add2_static oversubscription1 oversubscription2 oversubscription3 prefetch

add: add.cu
	nvcc add.cu -o add

add2_static: add2_static.cu
	nvcc add2_static.cu -o add2_static

oversubscription1: oversubscription1.cu
	nvcc -arch=sm_75 -O3 -DUNIFIED oversubscription1.cu -o oversubscription1

oversubscription2: oversubscription2.cu
	nvcc -arch=sm_75 -O3 oversubscription2.cu -o oversubscription2

oversubscription3: oversubscription3.cu
	nvcc -arch=sm_75 -O3 oversubscription3.cu -o oversubscription3

prefetch: prefetch.cu
	nvcc -arch=sm_60 -O3 prefetch.cu -o prefetch

.PHONY: clean

clean:
	rm -rf add add2_static oversubscription1 oversubscription2 oversubscription3 prefetch
