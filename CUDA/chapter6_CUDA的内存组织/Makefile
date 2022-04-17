all: static query

static: static.cu
	nvcc -O3 -DUSE_DP static.cu -o static

query: query.cu
	nvcc -O3 query.cu -o query
.PHONY: clean

clean:
	rm -rf static query
