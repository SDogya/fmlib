#!/bin/bash

# This environment variables needed
# to speed up the testing step
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

pytest "$@"
