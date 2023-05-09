cmake_minimum_required(VERSION 3.1)
project(Tracker)
set CMAKE_PREFIX_PATH /Users/arihanttripathi/Documents/C++ML/ML/libtorch
set (CMAKE_INSTALL_PREFIX /usr/local)
set(CMAKE_BUILD_TYPE Release)
set(OPENCV_EXTRA_MODULES_PATH /Users/arihanttripathi/Documents/C++ML/ML/opencv2.framework)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "OPENCV Lib Status:")
message(STATUS "config: ${OpenCV_DIR}")
message(STATUS "libraries: ${OpenCV_Libs}")s
message(STATUS "include path: $(OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})