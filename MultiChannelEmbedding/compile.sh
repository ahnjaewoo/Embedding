icc Embedding.cpp -o Embedding.out -std=c++11 -O3 -xHost -qopenmp -L/usr/local/lib -L/usr/include/armadillo_bits -DARMA_DONT_USE_WRAPPER -llapack -lopenblas -lboost_system -lboost_serialization
icc Test.cpp -o Test.out -std=c++11 -O3 -xHost -qopenmp -L/usr/local/lib -L/usr/include/armadillo_bits -DARMA_DONT_USE_WRAPPER -llapack -lopenblas -lboost_system -lboost_serialization
