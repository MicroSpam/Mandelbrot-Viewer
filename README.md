# Mandelbrot-Viewer
A realtime Mandelbrot set rendered in software written in C++20 using SDL2 and OpenMP.

Very simple, and not very optimal; achieves acceptable results on machines with lots of cores: preferably 4 or more.
Use the mouse to pan around, and the scroll wheel to zoom in and out.

This doesn't do anything fancy to extend floating point accuracy, so there is a limited amount of zoom
until you reach the limits of double precision IEEE-754. I did not attempt any fancy optimizations
either. I'm sure that the drawing routines could be sped up considerably, but I'm not currently 
interested in working on this any further.

