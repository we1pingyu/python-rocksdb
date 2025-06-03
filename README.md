```bash
make shared_lib -j
pip install pybind11 setuptools wheel
cd rocksdb_pybinding
python setup.py build_ext --inplace
pip install .
cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
```
