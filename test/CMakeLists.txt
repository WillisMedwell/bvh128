project(test)

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Download Catch2 Library.
set(Catch2_Header "catch.hpp")
set(Catch2_URL "https://github.com/catchorg/Catch2/releases/download/v2.13.8/catch.hpp")
set(Catch2_Download_Dir "${CMAKE_BINARY_DIR}/_deps/catch2")

file(MAKE_DIRECTORY ${Catch2_Download_Dir})

file(DOWNLOAD ${Catch2_URL} "${Catch2_Download_Dir}/${Catch2_Header}"
    EXPECTED_HASH SHA256=f5adf6a4c3237920421d51a11112904084043b015acf2c9058e8f36b92ab2921
    TIMEOUT 60 
    TLS_VERIFY ON
)

add_executable(test "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp")

target_include_directories(test PRIVATE "${CMAKE_BINARY_DIR}/_deps/")
target_include_directories(test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")