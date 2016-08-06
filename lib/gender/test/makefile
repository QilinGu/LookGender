#
#
#

OPENCV = `pkg-config opencv --cflags --libs`  \
          -I/usr/include/opencv  \

#
#
#

output:
	$(CXX) \
    test.cpp \
    -L../ -lGenderClassify -lrt $(OPENCV) -o gender -O3 -std=c++11
