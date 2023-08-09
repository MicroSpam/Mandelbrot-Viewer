#!/bin/sh
#g++ -std=c++20 -g -lSDL2 -fopenmp -march=native -lpthread mandelbrot.cpp -o mandelbrot
g++ -std=c++20 -O3 -fopenmp -march=native -lSDL2 -lpthread mandelbrot.cpp -o mandelbrot
