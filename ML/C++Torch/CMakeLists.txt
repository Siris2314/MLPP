cmake_minimum_required(VERSION 3.2.0)

project(C++Torch)

set(CMAKE_PREFIX_PATH /Users/arihanttripathi/Documents/C++ML/ML/libtorch)

find_package(Torch REQUIRED)

add_executable(main src/network.cpp
                    include/network.h)

target_include_directories(main PRIVATE include)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)