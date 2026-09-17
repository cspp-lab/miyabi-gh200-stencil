NVCC     ?= nvcc
MPICXX   ?= mpicxx
ARCH     ?= sm_90
CXXFLAGS := -O3 -std=c++17 -arch=$(ARCH) -ccbin=$(MPICXX) -Xcompiler -fopenmp
TARGET   := stencil3d
SRC      := src/stencil3d.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
