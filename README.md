# bvh128
A C++20 bounding-volume-heirarchy optimised with 128bit simd operations.

## Header Only & Light Weight
**Peformance of bvh128**

| Num of AABB | Time to construct | Time to Query | Time to Build and Query 1000 |
| ----------- | ----------------- | ------------- | ---------------------------- |
| 500         | 100 us            | 0.25 us       | 350 us                       |
| 1,000       | 200 us            | 0.30 us       | 500 us                       |
| 10,000      | 2,500 us          | 0.75 us       | 3,250 us                     |
| 100,000     | 30,000 us         | 1.00 us       | 31,000 us                    |
| 1,000,000   | 315,000 us        | 4.50 us       | 319,500 us                   |

**Performance of array** _(using simd as well for a fair test)_

| Num of AABB | Time to Query | Time to Query 1000 |
| ----------- | ------------- | ------------------ |
| 500         | 0.60 us       | 600 us             |
| 1,000       | 0.90 us       | 900 us             |
| 10,000      | 8.25 us       | 8,250 us           |
| 100,000     | 155.90 us     | 155,900 us         |
| 1,000,000   | 1513.75 us    | 1,513,750 us       |

**Conclusion** 
Depsite being quite simple, just using a flat array is quite performant and you pay no cost for building the tree. 
- In applications where the volumes are constantly updated and there are few queries, just use a contiguous array.
- However when doing many queries, a bvh128 offers performance improvements.

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




