cmake_minimum_required(VERSION 3.28)
project(NSU_FIT_OPP)

set(CMAKE_CXX_STANDARD 14)

include_directories(OPP_lab3)

add_executable(NSU_FIT_OPP OPP_lab5/main.cpp
        OPP_lab5/consts.h
        OPP_lab4/main.cpp)
