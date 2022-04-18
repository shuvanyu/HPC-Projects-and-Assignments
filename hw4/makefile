EXECUTABLES = dot_prod mat_vec_mul jacobi2d

all: $(EXECUTABLES)

dot_prod: dot_prod.cu
	nvcc -std=c++11  dot_prod.cu -o dot_prod

mat_vec_mul: mat_vec_mul.cu
	nvcc -std=c++11  mat_vec_mul.cu -o mat_vec_mul

jacobi2d: jacobi2d.cu
	nvcc -std=c++11  jacobi2d.cu -o jacobi2d

clean:
	rm -rf $(EXECUTABLES)