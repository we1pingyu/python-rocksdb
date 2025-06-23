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

    bool put(const py::bytes& key, const py::bytes& value) {
        if (!db) return false;
        std::string k = key;
        std::string v = value;
        rocksdb::Status status = db->Put(rocksdb::WriteOptions(), k, v);
        return status.ok();
    }

    py::object get(const py::bytes& key) {
        if (!db) return py::none();
        std::string k = key;
        std::string value;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), k, &value);
        if (!status.ok()) {
            return py::none();
        }
        return py::bytes(value);
    }

    py::dict multiget(const std::vector<py::bytes> &keys)
	{
		py::dict result;
		if (!db)
			return result;

		std::vector<rocksdb::Slice> slices;
		std::vector<std::string> key_storage;  // to hold memory for keys

		for (const auto &k : keys)
		{
			std::string key_str = static_cast<std::string>(k);
			key_storage.push_back(key_str);               // store copy
			slices.emplace_back(key_storage.back());      // make slice
		}

		std::vector<std::string> values;
		std::vector<rocksdb::Status> statuses = db->MultiGet(rocksdb::ReadOptions(), slices, &values);

		for (size_t i = 0; i < keys.size(); ++i)
		{
			if (statuses[i].ok())
			{
				result[keys[i]] = py::bytes(values[i]);  // return bytes
			}
			else
			{
				result[keys[i]] = py::none();
			}
		}

		return result;
	}

	bool delete_key(const py::bytes &key)
	{
		if (!db)
			return false;
		std::string key_str = static_cast<std::string>(key);
		rocksdb::Status status = db->Delete(rocksdb::WriteOptions(), key_str);
		return status.ok();
	}


    void set_custom_option(int value)
    {
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
        .def("multiget", &RocksDBWrapper::multiget)
        .def("delete", &RocksDBWrapper::delete_key)
        .def("set_custom_option", &RocksDBWrapper::set_custom_option);

    py::class_<rocksdb::Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("create_if_missing", &rocksdb::Options::create_if_missing)
        .def_readwrite("max_open_files", &rocksdb::Options::max_open_files);
}
