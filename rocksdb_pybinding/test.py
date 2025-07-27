#!/usr/bin/env python3
"""
测试脚本：测试RocksDB的batch_put和batch_get接口
"""

import os
import shutil
import torch
import numpy as np
import rocksdb_binding
from safetensor_helper import SafetensorHelper

def generate_test_kv_cache(batch_size=5, key_shape=(32, 64), value_shape=(32, 64)):
    """生成测试用的KV cache数据"""
    kv_caches = []
    for i in range(batch_size):
        # 生成随机的float16数据，范围控制在[-1, 1]之间以确保量化效果
        k = torch.randn(key_shape, dtype=torch.float16) * 0.5
        v = torch.randn(value_shape, dtype=torch.float16) * 0.5
        kv_caches.append((k, v))
    return kv_caches

def test_safetensor_helper():
    """测试SafetensorHelper的基本功能"""
    print("=== Testing SafetensorHelper ===")
    
    helper = SafetensorHelper("./test_storage")
    
    # 生成测试数据
    test_caches = generate_test_kv_cache(3)
    
    # 测试保存
    metadata = helper.save_kv_caches("test_file.safetensors", test_caches)
    print(f"Saved metadata: {metadata}")
    
    # 测试加载
    loaded_caches = helper.load_kv_caches("test_file.safetensors", [0, 1, 2])
    print(f"Loaded {len(loaded_caches)} caches")
    
    # 验证数据（考虑量化误差）
    for i, ((orig_k, orig_v), (loaded_k, loaded_v)) in enumerate(zip(test_caches, loaded_caches)):
        k_diff = torch.abs(orig_k - loaded_k).max().item()
        v_diff = torch.abs(orig_v - loaded_v).max().item()
        print(f"Cache {i}: key_diff={k_diff:.6f}, value_diff={v_diff:.6f}")
        
        # 检查误差是否在可接受范围内
        # 动态量化的误差应该更小，大约是原始值的1/127的精度
        max_orig_k = torch.abs(orig_k).max().item()
        max_orig_v = torch.abs(orig_v).max().item()
        
        k_tolerance = max(max_orig_k / 127.0, 1e-4)  # 至少1e-4的容差
        v_tolerance = max(max_orig_v / 127.0, 1e-4)
        
        assert k_diff < k_tolerance, f"Key difference too large: {k_diff} > {k_tolerance}"
        assert v_diff < v_tolerance, f"Value difference too large: {v_diff} > {v_tolerance}"
    
    print("SafetensorHelper test passed!")
    
    # 清理
    helper.cleanup_file("test_file.safetensors")

def test_batch_operations():
    """测试RocksDB的batch_put和batch_get操作"""
    print("\n=== Testing RocksDB Batch Operations ===")
    
    # 清理和准备目录
    db_path = "./test_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    storage_path = "./kv_cache_storage"
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)
    
    # 创建RocksDB实例
    db = rocksdb_binding.RocksDB()
    if not db.open(db_path):
        raise RuntimeError("Failed to open database")
    
    print("Database opened successfully")
    
    # 生成测试数据
    batch_size = 8
    kv_caches = generate_test_kv_cache(batch_size, key_shape=(16, 32), value_shape=(16, 32))
    
    # 生成keys
    keys = [f"test_key_{i}".encode() for i in range(batch_size)]
    
    print(f"Generated {batch_size} test KV caches")
    
    # 测试batch_put
    print("Testing batch_put...")
    success = db.batch_put(keys, kv_caches)
    if not success:
        raise RuntimeError("batch_put failed")
    print("batch_put successful")
    
    # 测试batch_get - 获取所有数据
    print("Testing batch_get (all keys)...")
    retrieved_caches = db.batch_get(keys)
    
    if len(retrieved_caches) != batch_size:
        raise RuntimeError(f"Expected {batch_size} results, got {len(retrieved_caches)}")
    
    print(f"Retrieved {len(retrieved_caches)} caches")
    
    # 验证数据完整性
    print("Verifying data integrity...")
    
    for i in range(batch_size):
        if retrieved_caches[i] is None:
            raise RuntimeError(f"Cache {i} is None")
        
        orig_k, orig_v = kv_caches[i]
        retr_k, retr_v = retrieved_caches[i]
        
        k_diff = torch.abs(orig_k - retr_k).max().item()
        v_diff = torch.abs(orig_v - retr_v).max().item()
        
        print(f"Cache {i}: key_diff={k_diff:.6f}, value_diff={v_diff:.6f}")
        
        # 动态计算容差
        max_orig_k = torch.abs(orig_k).max().item()
        max_orig_v = torch.abs(orig_v).max().item()
        
        k_tolerance = max(max_orig_k / 127.0, 1e-4)
        v_tolerance = max(max_orig_v / 127.0, 1e-4)
        
        assert k_diff < k_tolerance, f"Key difference too large for cache {i}: {k_diff} > {k_tolerance}"
        assert v_diff < v_tolerance, f"Value difference too large for cache {i}: {v_diff} > {v_tolerance}"
    
    print("Data integrity check passed!")
    
    # 测试部分获取
    print("Testing partial batch_get...")
    partial_keys = [keys[1], keys[3], keys[5]]
    partial_results = db.batch_get(partial_keys)
    
    if len(partial_results) != 3:
        raise RuntimeError(f"Expected 3 partial results, got {len(partial_results)}")
    
    # 验证部分结果
    expected_indices = [1, 3, 5]
    for i, (result, expected_idx) in enumerate(zip(partial_results, expected_indices)):
        if result is None:
            raise RuntimeError(f"Partial result {i} is None")
        
        orig_k, orig_v = kv_caches[expected_idx]
        retr_k, retr_v = result
        
        k_diff = torch.abs(orig_k - retr_k).max().item()
        v_diff = torch.abs(orig_v - retr_v).max().item()
        
        # 动态计算容差
        max_orig_k = torch.abs(orig_k).max().item()
        max_orig_v = torch.abs(orig_v).max().item()
        
        k_tolerance = max(max_orig_k / 127.0, 1e-4)
        v_tolerance = max(max_orig_v / 127.0, 1e-4)
        
        assert k_diff < k_tolerance, f"Partial key difference too large: {k_diff} > {k_tolerance}"
        assert v_diff < v_tolerance, f"Partial value difference too large: {v_diff} > {v_tolerance}"
    
    print("Partial batch_get test passed!")
    
    # 测试不存在的key
    print("Testing non-existent keys...")
    non_existent_keys = [b"non_existent_1", b"non_existent_2"]
    non_existent_results = db.batch_get(non_existent_keys)
    
    for i, result in enumerate(non_existent_results):
        if result is not None:
            print(f"Warning: Expected None for non-existent key {i}, got {type(result)}")
    
    print("Non-existent keys test completed")

def test_edge_cases():
    """测试边界情况"""
    print("\n=== Testing Edge Cases ===")
    
    db_path = "./test_edge_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    db = rocksdb_binding.RocksDB()
    if not db.open(db_path):
        raise RuntimeError("Failed to open edge case database")
    
    # 测试空输入
    print("Testing empty inputs...")
    try:
        db.batch_put([], [])
        print("Empty batch_put handled correctly")
    except Exception as e:
        print(f"Empty batch_put raised exception: {e}")
    
    # 测试不匹配的keys和values长度
    print("Testing mismatched lengths...")
    try:
        keys = [b"key1", b"key2"]
        caches = [generate_test_kv_cache(1)[0]]  # 只有一个cache
        db.batch_put(keys, caches)
        print("ERROR: Should have raised exception for mismatched lengths")
    except Exception as e:
        print(f"Correctly caught mismatched lengths: {e}")
    
    # 测试大数据
    print("Testing large data...")
    large_caches = generate_test_kv_cache(2, key_shape=(128, 256), value_shape=(128, 256))
    large_keys = [b"large_key_1", b"large_key_2"]
    
    success = db.batch_put(large_keys, large_caches)
    if success:
        print("Large data batch_put successful")
        
        retrieved = db.batch_get(large_keys)
        if len(retrieved) == 2 and all(r is not None for r in retrieved):
            print("Large data batch_get successful")
        else:
            print("Large data batch_get failed")
    else:
        print("Large data batch_put failed")

def main():
    """主测试函数"""
    print("Starting RocksDB Batch Operations Test Suite...")
    
    try:
        # 测试SafetensorHelper
        test_safetensor_helper()
        
        # 测试batch操作
        test_batch_operations()
        
        # 测试边界情况
        test_edge_cases()
        
        print("\n=== All Tests Passed! ===")
        
    except Exception as e:
        print(f"\n=== Test Failed: {e} ===")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试文件
        cleanup_paths = ["./test_db", "./test_edge_db", "./test_storage", "./kv_cache_storage"]
        for path in cleanup_paths:
            if os.path.exists(path):
                shutil.rmtree(path)
        print("Cleanup completed")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)