NVCC  ?= nvcc
ARCH  ?= sm_90
FLAGS := -O2 -std=c++17 -arch=$(ARCH)

all: smid work

smid: src/smid.cu
	$(NVCC) $(FLAGS) -o $@ $<

work: src/work.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f smid work

.PHONY: all clean
