#!/bin/sh

g++ -o velet.x -fopenmp md3leap.c
g++ -o leap.x -fopenmp md3velet.c

nvcc -o leapcu.x -I/usr/local/cuda-11.0/samples/common/inc md3leap.cu
nvcc -o veletcu.x -I/usr/local/cuda-11.0/samples/common/inc md3velet.cu
