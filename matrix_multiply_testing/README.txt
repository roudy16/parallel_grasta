# MATRIX MULTIPLICATION TESTING

Here I am experimenting with CUDA optimization techniques. I'm using a straight-
forward matrix multiplication that is conducted entirely on the CPU as a baseline
for comparison with my various GPU implementations of the same matrix multiplication.

As of 7 July 2015 For a 2048 by 2048 matrix the optimized gpu matrix multiplication 
executes approximately 200x faster than the CPU version. The optimized gpu version is 
only slightly faster than the naive gpu version, about 2-3% faster.

TODO: Add dependency info
