NVCC     ?= nvcc
MPICXX   ?= mpicxx
ARCH     ?= sm_90
ACC_GPU  ?= cc90
CXXFLAGS     := -O3 -std=c++17 -arch=$(ARCH) -ccbin=$(MPICXX) -Xcompiler -fopenmp
ACC_CXXFLAGS := -O3 -std=c++17 -acc=gpu -gpu=$(ACC_GPU) -mp -Minfo=accel

CUDA_TARGET := stencil3d
ACC_TARGET  := stencil3d_acc
CUDA_SRC    := src/stencil3d.cu
ACC_SRC     := src/stencil3d_acc.cpp

all: $(CUDA_TARGET) $(ACC_TARGET)

$(CUDA_TARGET): $(CUDA_SRC)
	$(NVCC) $(CXXFLAGS) -o $@ $<

$(ACC_TARGET): $(ACC_SRC)
	$(MPICXX) $(ACC_CXXFLAGS) -o $@ $<

clean:
	rm -f $(CUDA_TARGET) $(ACC_TARGET)

.PHONY: all clean
