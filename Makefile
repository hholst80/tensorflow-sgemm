OPENBLAS_FLAGS=$(shell pkg-config --cflags --libs blas-openblas)
TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
openblas/openblas.so: openblas/openblas.cxx
	g++ -std=c++11 -shared $^ -o $@ -fPIC -I $(TF_INC) $(OPENBLAS_FLAGS) -O2 -D_GLIBCXX_USE_CXX11_ABI=0
clean:
	$(RM) openblas/openblas.so
