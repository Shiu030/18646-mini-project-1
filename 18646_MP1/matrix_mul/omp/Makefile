PROJ = matrix_mul
CC = g++

CFLAGS = -c -g -Wall -fno-tree-vectorize -fno-inline-functions -I/opt/local/include -I$(HOME)/cppunit/include
LDFLAGS = -L/opt/local/lib -L$(HOME)/cppunit/lib
LIBS = -lcppunit -ldl
OBJS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OMPFLAGS = -fopenmp
SIMDFLAGS = -mno-avx -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -O1 -lm

all: $(PROJ)

$(PROJ): $(OBJS)
	$(CC) $(OMPFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o : %.cpp
	$(CC) $(OMPFLAGS) $(CFLAGS) $< -o $@ $(SIMDFLAGS)

%.o : %.cpp %.h
	$(CC) $(CFLAGS) $< -o $@ $(SIMDFLAGS)

clean:
	rm -f $(PROJ) $(OBJS) *.xml

test:
	./matrix_mul -i matrix_mul_02.dat
	cat omp_results.xml
