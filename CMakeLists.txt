cmake_minimum_required(VERSION 3.24)
project(polar_codes)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O2")

add_executable(polar_codes main.cpp
        random.h
        random.cpp
        support.h
        stack.cpp
        stack.h
        ChannelAWGN.cpp
        ChannelAWGN.h
        PolarCode.cpp
        PolarCode.h)
