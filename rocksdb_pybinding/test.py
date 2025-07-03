import rocksdb_binding
import os

# 创建数据库实例
db = rocksdb_binding.RocksDB()

# 打开数据库
if db.open(os.path.expanduser("~/test_db")):
    # 写入数据
    db.put(b"key1", b"value1")
    db.put(b"key2", b"value2")

    # 读取数据
    value = db.get(b"key1")

    print(f"key1: {value}")

    # 使用自定义选项
    db.set_custom_option(1000)

    # 删除键
    db.delete(b"key2")
    value = db.get(b"key2")
    print(value)
    
    values = db.multiget([b"key1", b"key2"])
    print(f"Multiget values: {values}")
    
else:
    print("Failed to open database")
