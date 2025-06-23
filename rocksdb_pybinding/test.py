import rocksdb_binding

# 创建数据库实例
db = rocksdb_binding.RocksDB()

# 打开数据库
if db.open("/tmp/test_db"):
    # 写入数据
    db.put("key1", "value1")
    db.put("key2", "value2")

    # 读取数据
    value = db.get("key1")

    print(f"key1: {value}")

    # 使用自定义选项
    db.set_custom_option(1000)

    # 删除键
    db.delete("key2")
    value = db.get("key2")
    print(value)
    
    values = db.multiget(["key1", "key2"])
    print(f"Multiget values: {values}")
    
else:
    print("Failed to open database")
