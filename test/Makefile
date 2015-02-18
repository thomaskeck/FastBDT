test: tests.cxx ../src/FBDT.cxx ../inc/FBDT.h
	g++ tests.cxx ../src/FBDT.cxx -o test -std=c++11 -I../inc/ -lgtest -L/usr/src/gtest/ -lpthread -ggdb3 -Wall

iris: main.cxx ../src/FBDT.cxx ../inc/FBDT.h
	g++ main.cxx ../src/FBDT.cxx -o iris -std=c++11 -I../inc/ -ggdb3 -Wall

pyinterface: pyinterface.cxx ../src/FBDT.cxx ../inc/FBDT.h
	g++ pyinterface.cxx -c -fPIC -std=c++11 -I../inc/  -lpthread -O3 -Wall -o pyinterface.o
	g++ ../src/FBDT.cxx -c -fPIC -std=c++11 -I../inc/  -lpthread -O3 -Wall -o FBDT.o
	g++ -shared -Wl,-soname,libPYFBDT.so -o libPYFBDT.so  pyinterface.o FBDT.o
