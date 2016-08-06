#
#
#

OPENCV = `pkg-config opencv --cflags --libs`  \
          -I/usr/include/opencv  \
          -Ilib/3rdparty/cv_ext  \
          -Ilib/3rdparty/piotr -Ilib/3rdparty/piotr/src  -Ilib/cf_libs/common \
          -Ilib/genderclass \
          -Ilib/cf_libs/kcf

#
#
#

output:
	$(CXX) \
    lib/3rdparty/piotr/src/gradientMex.o \
    lib/3rdparty/cv_ext/init_box_selector.cpp.o \
    lib/3rdparty/cv_ext/math_spectrums.cpp.o \
    lib/3rdparty/cv_ext/shift.cpp.o lib/cf_libs/common/math_helper.cpp.o \
    lookgender.cpp lib/pico/picornt.c \
    -L./lib/gender -lGenderClassify -lrt $(OPENCV) -o gender -O3 -std=c++11
