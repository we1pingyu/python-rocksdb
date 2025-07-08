cd ..
make shared_lib -j
cd rocksdb_pybinding
python setup.py build_ext --inplace
pip install .
python test.py