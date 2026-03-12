import numpy as np
import pickle
import torch

class FlightDataScaler:
    def __init__(self):
        """
        专门用于飞行数据的归一化器。
        支持 Z-Score, Min-Max, 和固定缩放 (Scale)。
        支持向量化操作，以便在 NeuralODE 中高效调用。
        """
        # 存储统计量: {'var_name': {'mean': ..., 'std': ..., 'method': ...}}
        self.stats = {}
        
        # 缓存线性变换系数 (用于加速向量化运算)
        # 结构: {'keys_tuple': (scale_tensor, offset_tensor)}
        self._cache_params = {}

        # 定义归一化配置
        self.config = {
            # --- 控制量 (Min-Max) ---
            'delta_e': 'minmax_sym', # [-1, 1]
            'delta_a': 'minmax_sym',
            'delta_r': 'minmax_sym',
            'delta_t': 'minmax_pos', # [0, 1]
            'rpm':     'minmax_pos',
            
            # --- 状态量 (Z-Score) ---
            'u': 'zscore', 'v': 'zscore', 'w': 'zscore',
            'p': 'zscore', 'q': 'zscore', 'r': 'zscore',
            'TAS': 'zscore', 'alpha': 'zscore', 'beta': 'zscore',
            'Q': 'zscore', 
            'phi': 'zscore', 'theta': 'zscore', 'psi': 'zscore',
            
            # --- 位置 (固定缩放) ---
            'x': 'scale_1000', # / 1000 (转为km级)
            'y': 'scale_1000',
            'z': 'scale_1000',
        }

    def fit(self, df):
        """计算并保存统计量 (Mean, Std, Min, Max)"""
        print("正在计算归一化统计参数...")
        for col in df.columns:
            if col not in self.config:
                continue
                
            method = self.config[col]
            data = df[col].values
            
            if method == 'zscore':
                self.stats[col] = {
                    'mean': np.mean(data),
                    'std': np.std(data) + 1e-6,
                    'method': method
                }
            elif method in ['minmax_sym', 'minmax_pos']:
                self.stats[col] = {
                    'min': np.min(data),
                    'max': np.max(data) + 1e-6,
                    'method': method
                }
            elif method == 'scale_1000':
                self.stats[col] = {
                    'factor': 1000.0,
                    'method': method
                }
        
        # 每次 fit 后清空缓存
        self._cache_params = {}
        return self

    def transform(self, df):
        """对 DataFrame 进行归一化 (用于预处理阶段)"""
        df_norm = df.copy()
        for col in df.columns:
            if col in self.stats:
                s = self.stats[col]
                method = s['method']
                val = df[col].values
                
                if method == 'zscore':
                    df_norm[col] = (val - s['mean']) / s['std']
                elif method == 'minmax_sym': # [-1, 1]
                    # 2 * (x - min) / (max - min) - 1
                    df_norm[col] = 2 * (val - s['min']) / (s['max'] - s['min']) - 1
                elif method == 'minmax_pos': # [0, 1]
                    df_norm[col] = (val - s['min']) / (s['max'] - s['min'])
                elif method == 'scale_1000':
                    df_norm[col] = val / s['factor']
                    
        return df_norm

    def _get_linear_coeffs(self, keys, device):
        """
        获取一组变量的线性变换系数 A 和 B。
        使得: x_real = x_norm * A + B
        """
        # 检查缓存
        cache_key = (tuple(keys), device)
        if cache_key in self._cache_params:
            return self._cache_params[cache_key]
        
        A_list = [] # 乘法系数 (Scale)
        B_list = [] # 加法系数 (Offset)
        
        for k in keys:
            if k not in self.stats:
                # 如果没有统计量，默认不变换 (A=1, B=0)
                A_list.append(1.0)
                B_list.append(0.0)
                continue
                
            s = self.stats[k]
            method = s['method']
            
            if method == 'zscore':
                # real = norm * std + mean
                A_list.append(s['std'])
                B_list.append(s['mean'])
                
            elif method == 'minmax_sym':
                # real = (norm + 1)/2 * (max-min) + min
                # real = norm * (max-min)/2 + [ (max-min)/2 + min ]
                span = s['max'] - s['min']
                scale = span / 2.0
                A_list.append(scale)
                B_list.append(scale + s['min'])
                
            elif method == 'minmax_pos':
                # real = norm * (max-min) + min
                span = s['max'] - s['min']
                A_list.append(span)
                B_list.append(s['min'])
                
            elif method == 'scale_1000':
                # real = norm * 1000
                A_list.append(s['factor'])
                B_list.append(0.0)

        # 转为 Tensor
        A_tensor = torch.tensor(A_list, dtype=torch.float32, device=device)
        B_tensor = torch.tensor(B_list, dtype=torch.float32, device=device)
        
        # 存入缓存
        self._cache_params[cache_key] = (A_tensor, B_tensor)
        return A_tensor, B_tensor

    def inverse_transform_vector(self, tensor_norm, keys):
        """
        向量化反归一化 (直接用于 ODEFunc forward)
        输入: (Batch, D) 归一化数据
        输出: (Batch, D) 真实物理数据
        """
        if not torch.is_tensor(tensor_norm):
            raise ValueError("此方法仅支持 Tensor 输入")
            
        A, B = self._get_linear_coeffs(keys, tensor_norm.device)
        
        # x_real = x_norm * A + B (利用广播机制)
        return tensor_norm * A + B

    def scale_derivative_vector(self, deriv_real, keys):
        """
        向量化导数归一化 (直接用于 ODEFunc forward return)
        输入: (Batch, D) 真实物理导数
        输出: (Batch, D) 归一化导数
        原理: dx_norm = dx_real / A
        """
        if not torch.is_tensor(deriv_real):
            raise ValueError("此方法仅支持 Tensor 输入")
            
        A, _ = self._get_linear_coeffs(keys, deriv_real.device)
        
        # dx_norm = dx_real / A
        return deriv_real / (A + 1e-9)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)
        print(f"Scaler 参数已保存至 {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
        self._cache_params = {} # 清空缓存
        print(f"Scaler 参数已加载")
        return self