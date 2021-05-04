cmake_minimum_required(VERSION 3.17)
project(MLP)

set(CMAKE_CXX_STANDARD 14)

add_executable(run mlp_ptr.h mlp_ptr.cpp)

