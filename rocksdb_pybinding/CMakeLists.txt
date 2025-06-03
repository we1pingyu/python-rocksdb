cmake_minimum_required(VERSION 3.12)
project(rocksdb_binding)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

# 设置RocksDB路径
set(ROCKSDB_PATH "..")

include_directories(${ROCKSDB_PATH}/include)
link_directories(${ROCKSDB_PATH})

pybind11_add_module(rocksdb_binding src/rocksdb_binding.cpp)
target_link_libraries(rocksdb_binding PRIVATE rocksdb)