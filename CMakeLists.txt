cmake_minimum_required(VERSION 3.16)

project(bvh128)

option(ENABLE_BVH128_TEST OFF)

if(ENABLE_BVH128_TEST)
    add_subdirectory(test) 
endif()
