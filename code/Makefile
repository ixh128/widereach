CC = gcc

#DEV = true
#CUDA = true #if you do not have a CUDA toolkit installed, comment out this line
#actually, CUDA makes it slower

ifdef DEV
  CFLAGS = -g -pg -Wall -fprofile-arcs -ftest-coverage -fsanitize=address
else
  CFLAGS = -O3 -DNDEBUG -funroll-loops
endif

INC = $(GUROBI_HOME)/include/
ifdef CUDA
  LIBS = -L$(GUROBI_HOME)/lib -L/usr/local/cuda/lib64 -lgurobi95 -lglpk -lm -lgsl -lnvblas -lopenblas
else
  LIBS = -L$(GUROBI_HOME)/lib -lgurobi95 -lglpk -lm -lgsl -lgslcblas -lcublas -L/usr/local/cuda/lib64 -lcudart
endif
#LIBS = -L$(GUROBI_HOME)/lib -lgurobi100 -lglpk -lm
BASE = widereach.h helper.h general.h glpk.h Makefile
OBJ = samples.o sparse_vector.o env.o indexing.o glpk.o params.o callback.o \
      iheur.o ibranch.o iselect.o irowgen.o solution_data.o ibingo.o \
      traversal.o signature.o children.o paths.o random.o hyperplane.o \
      obstruction.o interdiction.o cuts.o icutgen.o readsamples.o clusters.o \
      exec.o simplex.o \
      gurobi_exec.o gurobi.o bg_callback.o gurobi_callback.o constrained_hyperplane.o simulated_annealing.o sep_test.o cones.o priority_queue.o gurobi_cones.o siman_cones.o clustering.o greer_tree.o pointed_cones.o heapq.o cuda_fns.o mcts.o cone_exp.o eps_sample.o
UOBJ = utest.o

.PHONY: all coverage clean dependencies valgrind cleancoverage

all:
	echo $(CFLAGS); echo $(INC)

utest:	$(UOBJ) $(OBJ) $(BASE)
	rm --force *.gcno *gcda;\
	$(CC) $(CFLAGS) -o $@ $(UOBJ) $(OBJ) $(LIBS) -lcunit -fopenmp

coverage:	utest
	utest; gcovr
	
itest:	$(OBJ) $(BASE) itest.o
	$(CC) $(CFLAGS) -o $@ $(OBJ) itest.o $(LIBS) -fopenmp

experiment:	$(OBJ) $(BASE) experiment.o cleancoverage
	$(CC) $(CFLAGS) -o $@ $(OBJ) experiment.o $(LIBS) -fopenmp

hyperrnd:	$(OBJ) $(BASE) hyperrnd.o
	$(CC) $(CFLAGS) -o $@ $(OBJ) hyperrnd.o $(LIBS) -fopenmp
	
valgrind:	utest
	csh -c 'valgrind --tool=memcheck --leak-check=yes utest | & less -p Summary '

dependencies:
	sudo apt install libcunit1-dev glpk-doc libglpk-dev

cuda_fns.o: cuda_fns.cu $(BASE)
	nvcc -g -O3 -DNDEBUG -I $(INC) -lcublas -c -o $@ $<

%.o:	%.c $(BASE)
	$(CC) $(CFLAGS) -I $(INC) -c -o $@ $< -fopenmp
	
cleancoverage:
	rm --force *.gcno *gcda

clean:	cleancoverage
	rm --force *.o utest itest experiment experiment-* hyperrnd hyperrnd-*
