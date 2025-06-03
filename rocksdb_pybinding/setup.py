from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# RocksDB路径配置
ROCKSDB_PATH = ".."  # 修改为你的RocksDB路径

ext_modules = [
    Pybind11Extension(
        "rocksdb_binding",
        [
            "src/rocksdb_binding.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            f"{ROCKSDB_PATH}/include",
        ],
        libraries=["rocksdb"],
        library_dirs=[f"{ROCKSDB_PATH}"],
        language="c++",
        cxx_std=17,
    ),
]

setup(
    name="custom_rocksdb",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
