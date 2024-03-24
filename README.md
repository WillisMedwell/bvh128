# bvh128
A C++20 bounding-volume-heirarchy optimised with 128bit simd operations.

## Header Only & Light Weight

<details><summary><b>Adding to your Project</b></summary>

- Using CMake you can simply download it 
    _(this method means you dont need to compile it, unlike the fetch content module)._
    ```cmake
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps/bvh128")
    file(DOWNLOAD 
        "https://github.com/WillisMedwell/bvh128/releases/download/v0.1/bvh128.hpp"
        "${CMAKE_BINARY_DIR}/_deps/bvh128/bvh128.hpp"
        EXPECTED_HASH SHA256=e471cb629df17cbbfe405a8f42fd3f1cc1c8afc3c5b2debe699a144d084e82df
        TIMEOUT 5 TLS_VERIFY ON
    )
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${CMAKE_BINARY_DIR}/_deps/")
    ```
- Alternatively, just copy the header and add it to your include path!

</details>

<details><summary><b>Performance Review</b></summary>

- In applications where the volumes are constantly updated and there are few queries, just use a contiguous array.
- However when doing many queries, a bvh128 offers performance gains.

**bvh128 Performance**

| Num of AABB | Time to build bvh128 | Time to query bvh128 | Time to Build and Query 1000 |
| ----------- | -------------------- | -------------------- | ---------------------------- |
| 500         | 100 us               | 0.25 us              | 350 us                       |
| 1,000       | 200 us               | 0.30 us              | 500 us                       |
| 10,000      | 2,500 us             | 0.75 us              | 3,250 us                     |
| 100,000     | 30,000 us            | 1.00 us              | 31,000 us                    |
| 1,000,000   | 315,000 us           | 4.50 us              | 319,500 us                   |

**Control Performance** _(contiguous array using simd for a fair test)_

| Num of AABB | Time to find within array | Time to find 1000 |
| ----------- | ------------------------- | ----------------- |
| 500         | 0.60 us                   | 600 us            |
| 1,000       | 0.90 us                   | 900 us            |
| 10,000      | 8.25 us                   | 8,250 us          |
| 100,000     | 155.90 us                 | 155,900 us        |
| 1,000,000   | 1513.75 us                | 1,513,750 us      |

</details>

### Building Tests
- To build, run these commands after downloading the repo.

```cmd
cd bvh128
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_BVH128_TEST=true
cmake --build . --config Release
```




