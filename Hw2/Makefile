EXECUTABLES = val_test01_solved val_test02_solved MMult1 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp
COMPILER = g++
FLAGS = -fopenmp

all: $(EXECUTABLES)

val_test01_solved: val_test01_solved.cpp
	g++ -std=c++11 val_test01_solved.cpp -o val_test01_solved 

val_test02_solved: val_test02_solved.cpp
	g++ -std=c++11 val_test02_solved.cpp -o val_test02_solved 
	
MMult1: MMult1.cpp
	g++ -std=c++11 -fopenmp  -o3 -march=native MMult1.cpp -o MMult1

omp_solved2: omp_solved2.cpp
	g++ -std=c++11 -fopenmp omp_solved2.cpp -o omp_solved2

omp_solved3: omp_solved3.cpp
	g++ -std=c++11 -fopenmp omp_solved3.cpp -o omp_solved3

omp_solved4: omp_solved4.cpp
	g++ -std=c++11 -fopenmp omp_solved4.cpp -o omp_solved4

omp_solved5: omp_solved5.cpp
	g++ -std=c++11 -fopenmp omp_solved5.cpp -o omp_solved5

omp_solved6: omp_solved6.cpp
	g++ -std=c++11 -fopenmp omp_solved6.cpp -o omp_solved6

jacobi2D-omp: jacobi2D-omp.cpp
	g++ -std=c++11 -fopenmp -o3 jacobi2D-omp.cpp -o jacobi2D-omp

gs2D-omp: gs2D-omp.cpp
	g++ -std=c++11 -fopenmp -o3 gs2D-omp.cpp -o gs2D-omp

clean:
	rm -rf $(EXECUTABLES)