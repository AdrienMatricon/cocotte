cmake_minimum_required(VERSION 2.8)

project("Cocotte Learner")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

set (PURIST_FLAGS "-Wall -Wextra -Wcast-align -Wfloat-equal -Wformat=2 -Winit-self -Winline -Wmissing-declarations -Wmissing-format-attribute -Wmissing-include-dirs -Wpointer-arith -Wredundant-decls -Wshadow -Wunknown-pragmas -Wwrite-strings -Wno-long-long -Wno-unused-function -fPIE -fno-exceptions -fstack-protector --param=ssp-buffer-size=4 -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PURIST_FLAGS} -std=c++11 -frounding-math -fexceptions")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

include_directories(include)
include_directories(thirdparty)
add_subdirectory(src)
