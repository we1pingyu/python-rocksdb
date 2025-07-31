import torch
import numpy as np
from safetensors.torch import save_file, safe_open
import os
from pathlib import Path

class SafetensorHelper:
    def __init__(self, storage_dir="./kv_cache_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _quantize_tensor(self, tensor, scale_factor=None):
        """将float16 tensor量化到int8"""
        if scale_factor is None:
            # 动态计算缩放因子：使用tensor的最大绝对值
            max_val = torch.abs(tensor).max().item()
            if max_val == 0:
                scale_factor = 1.0
            else:
                scale_factor = 127.0 / max_val
        
        quantized = (tensor * scale_factor).clamp(-127, 127).round().to(torch.int8)
        return quantized, scale_factor
    
    def _dequantize_tensor(self, quantized_tensor, scale_factor):
        """将int8 tensor反量化到float16"""
        return (quantized_tensor.float() / scale_factor).to(torch.float16)
    
    def save_kv_caches(self, filename, kv_caches):
        """
        保存KV caches到safetensor文件
        Args:
            filename: 文件名
            kv_caches: list of tuples [(k1, v1), (k2, v2), ...]，每个k,v都是float16 tensor
        Returns:
            dict: 包含文件信息的metadata
        """
        if not kv_caches:
            raise ValueError("kv_caches cannot be empty")
        
        # 分别处理keys和values
        quantized_keys = []
        quantized_values = []
        key_scales = []
        value_scales = []
        
        for k_tensor, v_tensor in kv_caches:
            # 确保输入是float16类型的tensor
            if not isinstance(k_tensor, torch.Tensor) or not isinstance(v_tensor, torch.Tensor):
                raise ValueError("Each kv_cache item must be a tuple of (torch.Tensor, torch.Tensor)")
            
            # 转换为float16（如果不是的话）
            k_tensor = k_tensor.to(torch.float16)
            v_tensor = v_tensor.to(torch.float16)
            
            # 量化并保存缩放因子
            k_quantized, k_scale = self._quantize_tensor(k_tensor)
            v_quantized, v_scale = self._quantize_tensor(v_tensor)
            
            quantized_keys.append(k_quantized)
            quantized_values.append(v_quantized)
            key_scales.append(k_scale)
            value_scales.append(v_scale)
        
        # 将所有的keys和values分别concat成大tensor
        keys_tensor = torch.stack(quantized_keys, dim=0)  # shape: [num_caches, ...key_shape]
        values_tensor = torch.stack(quantized_values, dim=0)  # shape: [num_caches, ...value_shape]
        
        # 保存缩放因子
        key_scales_tensor = torch.tensor(key_scales, dtype=torch.float32)
        value_scales_tensor = torch.tensor(value_scales, dtype=torch.float32)
        
        # 保存到safetensor文件
        file_path = self.storage_dir / filename
        tensors_dict = {
            "keys": keys_tensor,
            "values": values_tensor,
            "key_scales": key_scales_tensor,
            "value_scales": value_scales_tensor,
            "num_caches": torch.tensor(len(kv_caches), dtype=torch.int32)
        }
        
        save_file(tensors_dict, str(file_path))
        
        return {
            "filename": filename,
            "num_caches": len(kv_caches),
            "keys_shape": list(keys_tensor.shape),
            "values_shape": list(values_tensor.shape)
        }
    
    def load_kv_caches(self, filename, offsets):
        """
        从safetensor文件加载KV caches
        Args:
            filename: 文件名
            offsets: list of int，要读取的cache索引
        Returns:
            list: [(k1, v1), (k2, v2), ...] 格式的反量化数据
        """
        file_path = self.storage_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        results = []
        
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            # 获取tensor slices和缩放因子
            keys_tensor_slice = f.get_slice("keys")
            values_tensor_slice = f.get_slice("values")
            key_scales_tensor = f.get_tensor("key_scales")
            value_scales_tensor = f.get_tensor("value_scales")
            
            # 按offsets读取数据
            for offset in offsets:
                # 读取第offset个cache
                k_quantized = keys_tensor_slice[offset]
                v_quantized = values_tensor_slice[offset]
                
                # 获取对应的缩放因子
                k_scale = key_scales_tensor[offset].item()
                v_scale = value_scales_tensor[offset].item()
                
                # 反量化
                k_dequantized = self._dequantize_tensor(k_quantized, k_scale)
                v_dequantized = self._dequantize_tensor(v_quantized, v_scale)
                
                results.append((k_dequantized, v_dequantized))
        
        return results
    
    def load_kv_caches_optimized(self, filename, offsets):
        """
        优化版本：合并连续的offsets来减少I/O操作
        """
        file_path = self.storage_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        if not offsets:
            return []
            
        # 对offsets排序并找到连续段
        sorted_offsets = sorted(enumerate(offsets), key=lambda x: x[1])
        
        results = [None] * len(offsets)  # 保持原始顺序
        
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            keys_tensor_slice = f.get_slice("keys")
            values_tensor_slice = f.get_slice("values")
            key_scales_tensor = f.get_tensor("key_scales")
            value_scales_tensor = f.get_tensor("value_scales")
            
            # 处理连续段
            i = 0
            while i < len(sorted_offsets):
                start_idx = i
                start_offset = sorted_offsets[i][1]
                
                # 找到连续段的结束
                while (i + 1 < len(sorted_offsets) and 
                       sorted_offsets[i + 1][1] == sorted_offsets[i][1] + 1):
                    i += 1
                
                end_offset = sorted_offsets[i][1]
                
                # 一次性读取连续段
                if start_offset == end_offset:
                    # 单个元素
                    k_quantized = keys_tensor_slice[start_offset]
                    v_quantized = values_tensor_slice[start_offset]
                    
                    k_scale = key_scales_tensor[start_offset].item()
                    v_scale = value_scales_tensor[start_offset].item()
                    
                    k_dequantized = self._dequantize_tensor(k_quantized, k_scale)
                    v_dequantized = self._dequantize_tensor(v_quantized, v_scale)
                    
                    orig_idx = sorted_offsets[start_idx][0]
                    results[orig_idx] = (k_dequantized, v_dequantized)
                else:
                    # 连续段
                    k_quantized_batch = keys_tensor_slice[start_offset:end_offset+1]
                    v_quantized_batch = values_tensor_slice[start_offset:end_offset+1]
                    k_scales_batch = key_scales_tensor[start_offset:end_offset+1]
                    v_scales_batch = value_scales_tensor[start_offset:end_offset+1]
                    
                    # 分别反量化每个元素（因为每个元素有不同的缩放因子）
                    for j in range(start_idx, i + 1):
                        batch_idx = j - start_idx
                        orig_idx = sorted_offsets[j][0]
                        
                        k_quantized = k_quantized_batch[batch_idx]
                        v_quantized = v_quantized_batch[batch_idx]
                        k_scale = k_scales_batch[batch_idx].item()
                        v_scale = v_scales_batch[batch_idx].item()
                        
                        k_dequantized = self._dequantize_tensor(k_quantized, k_scale)
                        v_dequantized = self._dequantize_tensor(v_quantized, v_scale)
                        
                        results[orig_idx] = (k_dequantized, v_dequantized)
                
                i += 1
        
        return results
    
    def cleanup_file(self, filename):
        """删除safetensor文件"""
        file_path = self.storage_dir / filename
        if file_path.exists():
            os.remove(file_path)
            return True
        return False
    
    def list_files(self):
        """列出所有存储的文件"""
        return [f.name for f in self.storage_dir.glob("*.safetensors")]