#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/status.h>

#include <atomic>
#include <iostream>
#include <sstream>

namespace py = pybind11;

class RocksDBWrapper {
 private:
  rocksdb::DB* db;
  rocksdb::Options options;
  static std::atomic<int> file_counter;  // 用于生成唯一文件名
  py::object safetensor_helper;          // Python helper object

 public:
  RocksDBWrapper() : db(nullptr) {
    options.create_if_missing = true;
    options.enable_blob_files = true;
    options.prepopulate_blob_cache = rocksdb::PrepopulateBlobCache::kFlushOnly;

    // Import Python helper
    py::module_ sys = py::module_::import("sys");
    py::module_ importlib = py::module_::import("importlib");
    py::module_ helper_module = py::module_::import("safetensor_helper");
    safetensor_helper = helper_module.attr("SafetensorHelper")();
  }

  ~RocksDBWrapper() {
    if (db) {
      delete db;
    }
  }

  bool open(const std::string& db_path) {
    std::cout << "C++ opening RocksDB at: " << db_path << std::endl;
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
    if (!status.ok()) {
      std::cout << "Failed to open RocksDB: " << status.ToString() << std::endl;
    }
    return status.ok();
  }

  bool put(const py::bytes& key, const py::bytes& value) {
    if (!db) throw std::runtime_error("Database not opened");
    std::string k = key;
    std::string v = value;
    rocksdb::Status status = db->Put(rocksdb::WriteOptions(), k, v);
    return status.ok();
  }

  py::object get(const py::bytes& key) {
    if (!db) throw std::runtime_error("Database not opened");
    std::string k = key;
    std::string value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), k, &value);
    if (!status.ok()) {
      return py::none();
    }
    return py::bytes(value);
  }

  bool probe(const py::bytes& key) {
    if (!db) throw std::runtime_error("Database not opened");
    std::string k = key;
    std::string value;
    auto read_options = rocksdb::ReadOptions();
    read_options.get_blob_value = false;
    rocksdb::Status status = db->Get(read_options, k, &value);
    if (status.IsNotFound()) {
      return false;
    } else if (status.ok()) {
      return true;
    } else {
      throw std::runtime_error("Error probing key: " + status.ToString());
    }
  }

  py::dict multiget(const std::vector<py::bytes>& keys) {
    py::dict result;
    if (!db) throw std::runtime_error("Database not opened");

    std::vector<rocksdb::Slice> slices;
    std::vector<std::string> key_storage(keys.size());

    for (const auto& k : keys) {
      std::string key_str = static_cast<std::string>(k);
      key_storage.push_back(key_str);
      slices.emplace_back(key_storage.back());
    }

    std::vector<std::string> values;
    auto column_families = std::vector<rocksdb::ColumnFamilyHandle*>(
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
          result[keys[i]] = py::bytes(value);
        }
      } else {
        result[keys[i]] = py::none();
      }
    }

    return result;
  }

  // 新增的batch_put方法
  bool batch_put(const std::vector<py::bytes>& keys,
                 const py::list& kv_caches) {
    if (!db) throw std::runtime_error("Database not opened");
    if (keys.size() != kv_caches.size()) {
      throw std::runtime_error("Keys and kv_caches must have the same length");
    }

    try {
      // 生成唯一文件名
      int file_id = file_counter.fetch_add(1);
      std::string filename =
          "kv_cache_" + std::to_string(file_id) + ".safetensors";

      // 调用Python helper来处理safetensor存储
      py::dict metadata =
          safetensor_helper.attr("save_kv_caches")(filename, kv_caches);

      // 将映射信息写入RocksDB
      rocksdb::WriteBatch batch;
      for (size_t i = 0; i < keys.size(); ++i) {
        std::string key_str = static_cast<std::string>(keys[i]);

        // 构造value: filename|offset
        std::string value_str = filename + "|" + std::to_string(i);

        batch.Put(key_str, value_str);
      }

      rocksdb::Status status = db->Write(rocksdb::WriteOptions(), &batch);
      return status.ok();

    } catch (const std::exception& e) {
      std::cerr << "Error in batch_put: " << e.what() << std::endl;
      return false;
    }
  }

  // 新增的batch_get方法
  py::list batch_get(const std::vector<py::bytes>& keys) {
    if (!db) throw std::runtime_error("Database not opened");

    try {
      // 从RocksDB获取映射信息
      std::vector<rocksdb::Slice> slices;
      std::vector<std::string> key_storage(keys.size());

      for (const auto& k : keys) {
        std::string key_str = static_cast<std::string>(k);
        key_storage.push_back(key_str);
        slices.emplace_back(key_storage.back());
      }

      std::vector<std::string> values;
      auto column_families = std::vector<rocksdb::ColumnFamilyHandle*>(
          slices.size(), db->DefaultColumnFamily());
      auto statuses = db->MultiGet(rocksdb::ReadOptions(), column_families,
                                   slices, &values, nullptr);

      // 组织文件和offset信息
      std::map<std::string, std::vector<int>>
          file_offsets;                 // filename -> list of offsets
      std::vector<int> result_indices;  // 保持结果顺序

      for (size_t i = 0; i < keys.size(); ++i) {
        if (statuses[i].ok()) {
          std::string value_str = values[i];
          size_t pipe_pos = value_str.find('|');
          if (pipe_pos != std::string::npos) {
            std::string filename = value_str.substr(0, pipe_pos);
            int offset = std::stoi(value_str.substr(pipe_pos + 1));

            file_offsets[filename].push_back(offset);
            result_indices.push_back(i);
          }
        } else {
          result_indices.push_back(-1);  // 标记未找到
        }
      }

      // 调用Python helper来读取数据
      py::list results = py::list();
      for (size_t i = 0; i < keys.size(); ++i) {
        results.append(py::none());
      }

      for (const auto& file_entry : file_offsets) {
        const std::string& filename = file_entry.first;
        const std::vector<int>& offsets = file_entry.second;

        py::list offset_list = py::cast(offsets);
        py::list loaded_caches =
            safetensor_helper.attr("load_kv_caches")(filename, offset_list);

        // 将结果放到正确位置
        int loaded_idx = 0;
        for (size_t i = 0; i < result_indices.size(); ++i) {
          if (result_indices[i] >= 0) {
            // 检查这个索引对应的文件是否匹配
            std::string value_str = values[result_indices[i]];
            size_t pipe_pos = value_str.find('|');
            if (pipe_pos != std::string::npos) {
              std::string current_filename = value_str.substr(0, pipe_pos);
              if (current_filename == filename) {
                results[result_indices[i]] = loaded_caches[loaded_idx++];
              }
            }
          }
        }
      }

      return results;

    } catch (const std::exception& e) {
      std::cerr << "Error in batch_get: " << e.what() << std::endl;
      return py::list();
    }
  }

  bool batch_put_original(const std::vector<py::bytes>& keys,
                          const std::vector<py::bytes>& values) {
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

  bool delete_key(const py::bytes& key) {
    if (!db) throw std::runtime_error("Database not opened");
    std::string key_str = static_cast<std::string>(key);
    rocksdb::Status status = db->Delete(rocksdb::WriteOptions(), key_str);
    return status.ok();
  }

  void set_custom_option(int value) { options.max_open_files = value; }
};

// 静态成员初始化
std::atomic<int> RocksDBWrapper::file_counter{0};

PYBIND11_MODULE(rocksdb_binding, m) {
  m.doc() = "Custom RocksDB Python binding with KV Cache support";

  py::class_<RocksDBWrapper>(m, "RocksDB")
      .def(py::init<>())
      .def("open", &RocksDBWrapper::open)
      .def("put", &RocksDBWrapper::put)
      .def("get", &RocksDBWrapper::get)
      .def("multiget", &RocksDBWrapper::multiget)
      .def("delete", &RocksDBWrapper::delete_key)
      .def("probe", &RocksDBWrapper::probe)
      .def("batch_put", &RocksDBWrapper::batch_put)  // 新的KV cache版本
      .def("batch_get", &RocksDBWrapper::batch_get)  // 新的KV cache版本
      .def("batch_put_original", &RocksDBWrapper::batch_put_original)  // 原版本
      .def("set_custom_option", &RocksDBWrapper::set_custom_option);

  py::class_<rocksdb::Options>(m, "Options")
      .def(py::init<>())
      .def_readwrite("create_if_missing", &rocksdb::Options::create_if_missing)
      .def_readwrite("max_open_files", &rocksdb::Options::max_open_files);
}