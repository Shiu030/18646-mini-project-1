# vim:set ts=8 sw=8 sts=0 noet:

#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    File:         Makefile                                                  */
#    Description:  Makefile for programs running a simple k-means clustering */
#                  algorithm                                                 */
#                                                                            */
#    Author:  Wei-keng Liao                                                  */
#             ECE Department Northwestern University                         */
#             email: wkliao@ece.northwestern.edu                             */
#    Copyright, 2005, Wei-keng Liao                                          */
#                                                                            */
#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

# Copyright (c) 2005 Wei-keng Liao
# Copyright (c) 2011 Serban Giuroiu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# ------------------------------------------------------------------------------

.KEEP_STATE:

all: seq omp

DFLAGS      =
OPTFLAGS    = -O1 -g -pg -fno-tree-vectorize -fno-inline-functions -mno-avx -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -lm
INCFLAGS    = -I.
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS)
LDFLAGS     = $(OPTFLAGS)
LIBS        =

OMPFLAGS    = -fopenmp

CC          = gcc

.c.o:
	$(CC) $(CFLAGS) -c $<

H_FILES     = kmeans.h

#------   OpenMP version -----------------------------------------
OMP_SRC     = omp_main.c \
	      omp_kmeans.c

OMP_OBJ     = $(OMP_SRC:%.c=%.o)

omp_kmeans.o: omp_kmeans.c $(H_FILES)
	$(CC) $(CFLAGS) $(OMPFLAGS) -c omp_kmeans.c

omp: omp_main
omp_main: $(OMP_OBJ) file_io.o
	$(CC) $(LDFLAGS) $(OMPFLAGS) -o omp_main $(OMP_OBJ) file_io.o $(LIBS)

#------   sequential version -----------------------------------------
SEQ_SRC     = seq_main.c   \
              seq_kmeans.c \
	      file_io.c    \
	      wtime.c

SEQ_OBJ     = $(SEQ_SRC:%.c=%.o)

$(SEQ_OBJ): $(H_FILES)

seq: seq_main
seq_main: $(SEQ_OBJ) $(H_FILES)
	$(CC) $(LDFLAGS) -o seq_main $(SEQ_OBJ) $(LIBS)

#---------------------------------------------------------------------
clean:
	rm -rf *.o omp_main seq_main \
               *.cluster_centres *.membership \
