# RUN for C++ 11                                                                                                                                                                                                                                                                
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

include_directories(${BOOST_INCLUDE_DIR})

SET(SOURCE
    src/gender_classification.cpp
    src/histogram.cpp
    src/lbp.cpp
)

SET(HEADERS
    includes/gender_classification.h
    includes/histogram.hpp
    includes/lbp.hpp
)
include_directories(./includes)

# add_library( FacePico STATIC ${SOURCE} ${HEADERS})
add_library( GenderClassify SHARED ${SOURCE} ${HEADERS})

install (TARGETS GenderClassify DESTINATION bin)
install (FILES HEADERS DESTINATION include)

