# TensorFlow BLAS matmul op

Building (on Ubuntu 16.04 LTS)

    $ sudo apt install libopenblas-dev

Build the binary library `openblas/openblas.so`

    $ make

Basic test:

    $ python basictest.py

Single thread performance test:

    $ env OMP_NUM_THREADS=1 python perftest.py

Install PIP repo

    $ pip install -e .
