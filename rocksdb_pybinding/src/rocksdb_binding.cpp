#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/status.h>

namespace py = pybind11;

class RocksDBWrapper
{
private:
    rocksdb::DB *db;
    rocksdb::Options options;

public:
    RocksDBWrapper() : db(nullptr)
    {
        options.create_if_missing = true;
    }

    ~RocksDBWrapper()
    {
        if (db)
        {
            delete db;
        }
    }

    bool open(const std::string &db_path)
    {
        rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
        return status.ok();
    }

    bool put(const std::string &key, const std::string &value)
    {
        if (!db)
            return false;
        rocksdb::Status status = db->Put(rocksdb::WriteOptions(), key, value);
        return status.ok();
    }

    std::string get(const std::string &key)
    {
        if (!db)
            return "";
        std::string value;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), key, &value);
        return status.ok() ? value : "";
    }

    bool delete_key(const std::string &key)
    {
        if (!db)
            return false;
        rocksdb::Status status = db->Delete(rocksdb::WriteOptions(), key);
        return status.ok();
    }

    // 添加你自定义的方法
    void set_custom_option(int value)
    {
        // 这里可以调用你修改的RocksDB内部代码
        options.max_open_files = value;
    }
};

PYBIND11_MODULE(rocksdb_binding, m)
{
    m.doc() = "Custom RocksDB Python binding";

    py::class_<RocksDBWrapper>(m, "RocksDB")
        .def(py::init<>())
        .def("open", &RocksDBWrapper::open)
        .def("put", &RocksDBWrapper::put)
        .def("get", &RocksDBWrapper::get)
        .def("delete", &RocksDBWrapper::delete_key)
        .def("set_custom_option", &RocksDBWrapper::set_custom_option);

    // 绑定RocksDB的原生类型（可选）
    py::class_<rocksdb::Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("create_if_missing", &rocksdb::Options::create_if_missing)
        .def_readwrite("max_open_files", &rocksdb::Options::max_open_files);
}