#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/status.h>

#include <iostream>

namespace py = pybind11;

class RocksDBWrapper {
 private:
  rocksdb::DB *db;
  rocksdb::Options options;

 public:
  RocksDBWrapper(bool blobdb = false) : db(nullptr) {
    options.create_if_missing = true;
    if (blobdb) {
      options.enable_blob_files = true;
      options.prepopulate_blob_cache =
          rocksdb::PrepopulateBlobCache::kFlushOnly;
    }
  }

  ~RocksDBWrapper() {
    if (db) {
      delete db;
    }
  }

  bool open(const std::string &db_path) {
    std::cout << "C++ opening RocksDB at: " << db_path << std::endl;
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
    if (!status.ok()) {
      std::cout << "Failed to open RocksDB: " << status.ToString() << std::endl;
    }
    return status.ok();
  }

  bool put(const py::bytes &key, const py::bytes &value) {
    if (!db) throw std::runtime_error("Database not opened");
    std::string k = key;
    std::string v = value;
    rocksdb::Status status = db->Put(rocksdb::WriteOptions(), k, v);
    return status.ok();
  }

  py::object get(const py::bytes &key) {
    if (!db) throw std::runtime_error("Database not opened");
    std::string k = key;
    std::string value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), k, &value);
    if (!status.ok()) {
      return py::none();
    }
    return py::bytes(value);
  }

  bool probe(const py::bytes &key) {
    if (!db) throw std::runtime_error("Database not opened");
    std::string k = key;
    std::string value;
    auto read_options = rocksdb::ReadOptions();
    read_options.get_blob_value = false;
    rocksdb::Status status = db->Get(read_options, k, &value);
    if (status.IsNotFound()) {
      return false;  // Key does not exist
    } else if (status.ok()) {
      return true;  // Key exists
    } else {
      throw std::runtime_error("Error probing key: " + status.ToString());
    }
  }

  py::dict multiget(const std::vector<py::bytes> &keys) {
    py::dict result;
    if (!db) throw std::runtime_error("Database not opened");

    std::vector<rocksdb::Slice> slices;
    std::vector<std::string> key_storage(keys.size());

    for (const auto &k : keys) {
      std::string key_str = static_cast<std::string>(k);
      key_storage.push_back(key_str);           // store copy
      slices.emplace_back(key_storage.back());  // make slice
    }
    std::vector<std::string> values;
    auto column_families = std::vector<rocksdb::ColumnFamilyHandle *>(
        slices.size(), db->DefaultColumnFamily());
    auto statuses = db->MultiGet(rocksdb::ReadOptions(), column_families,
                                 slices, &values, nullptr);

    for (size_t i = 0; i < keys.size(); ++i) {
      auto value = values[i];
      if (statuses[i].ok()) {
        if (value.size() == 0) {
          throw std::runtime_error("Empty value for key: " +
                                   static_cast<std::string>(keys[i]));
        } else {
          // Convert to bytes and store in result
          result[keys[i]] = py::bytes(value);
        }
      } else {
        result[keys[i]] = py::none();
      }
    }

    return result;
  }

  bool batch_put(const std::vector<py::bytes> &keys,
                 const std::vector<py::bytes> &values) {
    if (!db) throw std::runtime_error("Database not opened");
    if (keys.size() != values.size()) {
      throw std::runtime_error("Keys and values must have the same length");
    }

    rocksdb::WriteBatch batch;
    for (size_t i = 0; i < keys.size(); ++i) {
      std::string k = static_cast<std::string>(keys[i]);
      std::string v = static_cast<std::string>(values[i]);
      batch.Put(k, v);
    }

    rocksdb::Status status = db->Write(rocksdb::WriteOptions(), &batch);
    return status.ok();
  }

  bool delete_key(const py::bytes &key) {
    if (!db) throw std::runtime_error("Database not opened");
    std::string key_str = static_cast<std::string>(key);
    rocksdb::Status status = db->Delete(rocksdb::WriteOptions(), key_str);
    return status.ok();
  }

  void set_custom_option(int value) { options.max_open_files = value; }
};

PYBIND11_MODULE(rocksdb_binding, m) {
  m.doc() = "Custom RocksDB Python binding";

  py::class_<RocksDBWrapper>(m, "RocksDB")
      .def(py::init<>())
      .def("open", &RocksDBWrapper::open)
      .def("put", &RocksDBWrapper::put)
      .def("get", &RocksDBWrapper::get)
      .def("multiget", &RocksDBWrapper::multiget)
      .def("delete", &RocksDBWrapper::delete_key)
      .def("probe", &RocksDBWrapper::probe)
      .def("batch_put", &RocksDBWrapper::batch_put)
      .def("set_custom_option", &RocksDBWrapper::set_custom_option);

  py::class_<rocksdb::Options>(m, "Options")
      .def(py::init<>())
      .def_readwrite("create_if_missing", &rocksdb::Options::create_if_missing)
      .def_readwrite("max_open_files", &rocksdb::Options::max_open_files);
}
