all: reduce warp reduce1parallelism reduce2static

reduce: reduce.cu
	nvcc -g -arch=sm_50 reduce.cu -o reduce

warp: warp.cu
	nvcc -g -arch=sm_50 warp.cu -o warp

reduce1parallelism: reduce1parallelism.cu
	nvcc -g -arch=sm_50 reduce1parallelism.cu -o reduce1parallelism

reduce2static: reduce2static.cu
	nvcc -g -arch=sm_50 reduce2static.cu -o reduce2static

.PHONY: clean

clean:
	rm -rf reduce warp reduce1parallelism reduce2static
