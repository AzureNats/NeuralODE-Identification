import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from NeuralODEFunc import CoefficientNet
from NeuralODEFunc import AerialSystemODE
from flight_scaler import FlightDataScaler
from scipy.signal import savgol_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StaticAeroDataset(Dataset):
    def __init__(self, csv_path, scaler_path, props):
        print(f"正在加载静态数据集: {csv_path}")
        df = pd.read_csv(csv_path)
        scaler = FlightDataScaler()
        scaler.load(scaler_path)
        
        # 虚拟物理参数
        virtual_props = props.copy()
        virtual_props.update({
            'm': 420.0,
            'I': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'T_offset': [0, 0, 0]
        })
        
        # 实例化 ODE (仅用于调用calculate_thrust)
        virtual_ode = AerialSystemODE(
            neural_net=CoefficientNet(), 
            known_props=virtual_props, 
            scaler=scaler, 
            device=device
        )
        
        # 提取油门指令并计算真实推力
        power_col = 'rpm' if 'rpm' in df.columns else 'delta_t'
        delta_t_tensor = torch.tensor(df[power_col].values, dtype=torch.float32)
        thrust_np = virtual_ode.calculate_thrust(delta_t_tensor).detach().numpy()
        
        # 核心剥离公式：M_pure_aero = M_csv + 0.75 * Thrust
        df['M'] = df['M'].values + 0.75 * thrust_np

        # 1. 提取物理常数
        S = props['S']
        b = props['b']
        c = props['c']
        
        # 2. 数据平滑滤波 (气动力、力矩、气流角、角速度、空速)
        cols_to_filter = ['FX', 'FY', 'FZ', 'L', 'M', 'N', 
                          'TAS', 'alpha', 'beta', 'p', 'q', 'r']
                          
        for col in cols_to_filter:
            window = min(11, len(df) - (len(df) % 2 == 0)) 
            df[col] = savgol_filter(df[col].values, window_length=window, polyorder=2)

        # 3.风轴气动力转换至体轴
        fx_wind = df['FX'].values
        fy_wind = df['FY'].values
        fz_wind = df['FZ'].values
        
        alpha = df['alpha'].values
        beta = df['beta'].values
        
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # 使用 R_bw 矩阵进行投影
        fx_body = fx_wind * cos_a * cos_b - fy_wind * cos_a * sin_b - fz_wind * sin_a
        fy_body = fx_wind * sin_b + fy_wind * cos_b
        fz_body = fx_wind * sin_a * cos_b - fy_wind * sin_a * sin_b + fz_wind * cos_a

        # 4. 补齐缺失的状态量列
        df['u'] = df['TAS'] * cos_a * cos_b
        df['v'] = df['TAS'] * sin_b
        df['w'] = df['TAS'] * sin_a * cos_b
        
        df['x'] = 0.0
        df['y'] = 0.0
        df['z'] = 0.0

        # 4. 计算真实气动系数
        Q = df['Q'].values + 1e-6
        Cx_true = fx_body / (Q * S)
        Cy_true = fy_body / (Q * S)
        Cz_true = fz_body / (Q * S)
        Cl_true = df['L'].values / (Q * S * b)
        Cm_true = df['M'].values / (Q * S * c)
        Cn_true = df['N'].values / (Q * S * b)
        
        self.gt_coeffs = torch.tensor(
            np.stack([Cx_true, Cy_true, Cz_true, Cl_true, Cm_true, Cn_true], axis=1), 
            dtype=torch.float32
        )
        
        # 5. 提取并归一化网络输入 [u, v, w, p, q, r] 和 [delta_t, e, a, r]
        df_norm = scaler.transform(df)
        
        # 提取归一化后的状态量 (只需要前 6 个核心状态)
        state_keys_needed = ['u', 'v', 'w', 'p', 'q', 'r']
        state_norm = torch.tensor(df_norm[state_keys_needed].values, dtype=torch.float32)
        
        # 提取归一化后的控制量 (兼容 rpm 和 delta_t)
        power_col = 'rpm' if 'rpm' in df.columns else 'delta_t'
        control_keys = ['delta_e', 'delta_a', 'delta_r', power_col]
        control_norm = torch.tensor(df_norm[control_keys].values, dtype=torch.float32)
        
        # 构建网络最终输入 (状态 6 维 + 控制 4 维 = 10 维)
        self.nn_inputs = torch.cat([state_norm, control_norm], dim=1)
        
    def __len__(self):
        return len(self.nn_inputs)
        
    def __getitem__(self, idx):
        return self.nn_inputs[idx], self.gt_coeffs[idx]

def pretrain():
    props = {
        'S': 18.825,                
        'b': 9.804,                 
        'c': 1.932,                 
    }
    
    # 路径配置
    csv_path = 'Document41.csv'
    scaler_path = 'scaler41.pkl'
    model_save_path = 'pretrained_coeffs.pth'
    
    dataset = StaticAeroDataset(csv_path, scaler_path, props)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 实例化网络
    net = CoefficientNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    epochs = 200
    print("开始静态气动监督预训练...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred_y = net(batch_x)
            loss = loss_fn(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss (MSE): {avg_loss:.6e}")
            
    torch.save(net.state_dict(), model_save_path)
    print(f"预训练完成！权重已保存至: {model_save_path}")

if __name__ == '__main__':
    pretrain()