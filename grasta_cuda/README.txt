#GRASTA CUDA

This is my work in progress of a CUDA implementation of GRASTA. My approach has
been to identify sections that can be translated into CUDA kernels, write those
kernels, then replace the original sections of code with the new kernels. This way
I always have a working version of the program and can measure the performance gains
for each new iteration.

TODO: Provide good information about dependencies.
