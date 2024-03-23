# bvh128
A C++20 bounding-volume-heirarchy optimised with 128bit simd operations.

## Header Only & Light Weight
Peformance Notes:

### Building Tests
- To build, run these commands after downloading the repo.
- Uses Catch2 for testing _(no need to install it; cmake will download it and verify the hash)._
```cmd
cd bvh128
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_BVH128_TEST=true
cmake --build . --config Release
```




