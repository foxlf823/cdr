cc=g++

#cflags = -O0 -g3 -w \
	-I/home/fox/eclipse/workspace/FoxUtil \
	-I/usr/include/libxml2 \
	-msse3 -I /home/fox/Downloads/mshadow-master/mshadow -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 \
	-I/home/fox/Downloads/LibN3L-master -DUSE_CUDA=0 \
	
cflags = -O3 -w \
	-I/home/fox/eclipse/workspace/FoxUtil \
	-I/usr/include/libxml2 \
	-msse3 -I /home/fox/Downloads/mshadow-master/mshadow -DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 \
	-I/home/fox/Downloads/LibN3L-master -DUSE_CUDA=0 \
	-static-libgcc -static-libstdc++\

libs = -lm -lopenblas -lxml2 -Wl,-rpath,./

all: cdr

cdr: cdr.cpp NNcdr.h utils.h Classifier.h Attention.h EightLayer.h ClassifierWithoutEntity.h
	$(cc) -o cdr cdr.cpp $(cflags) $(libs)

	




clean:
	rm -rf *.o
	rm -rf cdr

