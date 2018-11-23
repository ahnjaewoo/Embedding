icc preprocess.cpp -o preprocess.out -std=c++11 -O3 -xHost -qopenmp -L/usr/local/lib -L/usr/include/armadillo_bits -DARMA_DONT_USE_WRAPPER -llapack -lopenblas -lboost_system -lboost_serialization
