CXX=g++

OPT = -I lib -O2 -mcmodel=medium -fopenmp -w 

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: P-Tucker 

P-Tucker: P-Tucker.cpp 
	$(CXX) $(CXXFLAGS)  -o $@  $<

demo: P-Tucker.cpp
	g++ -I lib -o P-Tucker P-Tucker.cpp -O2 -fopenmp -w -mcmodel=medium
	./P-Tucker datasets/chars_train_seed0.txt results/chars/80_0.01_0 3 80 35 


.PHONY: clean

clean:
	rm -f P-Tucker

