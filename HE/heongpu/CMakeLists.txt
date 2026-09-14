cmake_minimum_required(VERSION 3.26)
project(heongpu_smoke LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(HEonGPU REQUIRED)

add_executable(smoke_test src/smoke_test.cpp)
target_link_libraries(smoke_test PRIVATE HEonGPU::heongpu CUDA::cudart)
set_target_properties(smoke_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(bench_regular_bts src/bench_regular_bts.cpp)
target_link_libraries(bench_regular_bts PRIVATE HEonGPU::heongpu CUDA::cudart)
set_target_properties(bench_regular_bts PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
