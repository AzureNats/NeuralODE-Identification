import os
# ==========================================
# 1. 修复 OpenMP 冲突错误 (必须放在最前面)
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_full_test_data(filename="flight_data_preprocessing_test.csv"):
    """
    生成一个"脏"数据集，用于测试预处理管道的健壮性。
    包含：全通道数据、噪声、偏置、以及特殊的Psi跳变现象。
    """
    
    # 基础设置
    freq = 50.0
    dt = 1.0 / freq
    duration = 30.0
    time = np.arange(0, duration, dt)
    n = len(time)
    
    data = {'time': time}

    # ==============================
    # A. 控制量 (制造不同频率的信号)
    # ==============================
    # 1. 升降舵: 扫频信号 (Chirp) - 用于测试频域分析
    f_start, f_end = 0.1, 2.5
    k = (f_end - f_start) / duration
    chirp_phase = 2 * np.pi * (f_start * time + (k / 2) * time**2)
    data['delta_e'] = 5.0 * np.sin(chirp_phase) # +/- 5度
    
    # 2. 副翼: 简单的低频正弦波
    data['delta_a'] = 3.0 * np.sin(2 * np.pi * 0.5 * time)
    
    # 3. 方向舵: 阶梯信号
    data['delta_r'] = np.where(time > 15, 2.0, 0.0) # 15秒后蹬舵2度
    
    # 4. 油门 & 转速
    data['delta_t'] = 0.6 + 0.1 * np.sin(2 * np.pi * 0.1 * time)
    data['RPM'] = 5000 + 500 * data['delta_t'] + np.random.normal(0, 10, n) # 加点震动噪声

    # ==============================
    # B. 状态量 (不追求物理完美，只追求波形特征)
    # ==============================
    
    # --- 纵向 (Longitudinal) ---
    # 简单的滞后响应
    data['theta'] = np.deg2rad(data['delta_e'] * 0.8) # 俯仰角跟随升降舵
    data['q'] = np.gradient(data['theta'], dt)        # q 是 theta 的导数
    data['u_g'] = 30.0 + 2.0 * np.sin(0.2 * time)     # 速度波动
    data['w_g'] = data['u_g'] * np.tan(data['theta']) # 简单的垂向速度关系
    
    # --- 横向 (Lateral) ---
    data['phi'] = np.deg2rad(data['delta_a'] * 1.5)   # 滚转角跟随副翼
    data['p'] = np.gradient(data['phi'], dt)          # p 是 phi 的导数
    data['v_g'] = 5.0 * np.sin(0.5 * time)            # 侧向速度
    
    # --- 航向 (Directional) & Psi 跳变 ---
    # 模拟飞机以 10 deg/s 的速率持续右转
    # 初始角度设为 160度，这样过2秒就会碰到 180度边界
    yaw_rate_deg = 10.0 
    true_yaw_deg = 160.0 + yaw_rate_deg * time 
    
    # *** 关键点：制造 [-pi, pi] 的跳变 ***
    # 将连续角度映射到 -180 到 180 范围
    wrapped_yaw_deg = (true_yaw_deg + 180) % 360 - 180
    data['psi'] = np.deg2rad(wrapped_yaw_deg) # 存入弧度，这里会有明显的跳变
    
    data['r'] = np.deg2rad(yaw_rate_deg) * np.ones(n) # 偏航角速度是常数
    
    # --- 位置 ---
    data['lat'] = 39.9 + 0.001 * time # 简单的经纬度漂移
    data['lon'] = 116.4 + 0.001 * time
    data['h'] = 1000.0 - np.cumsum(data['w_g'] * dt) # 高度积分

    # ==============================
    # C. IMU 数据 (包含重力 + 噪声 + 偏置)
    # ==============================
    g = 9.81
    
    # 1. 运动加速度 (由速度导数估算)
    ax_kin = np.gradient(data['u_g'], dt)
    ay_kin = np.gradient(data['v_g'], dt)
    az_kin = np.gradient(data['w_g'], dt)
    
    # 2. 重力分量投影 (NED系 -> 机体系)
    # 简化公式，不搞复杂的旋转矩阵，只为了大概像样
    # g_x = -g * sin(theta)
    # g_y = g * sin(phi) * cos(theta)
    # g_z = g * cos(phi) * cos(theta)
    
    gx_proj = -g * np.sin(data['theta'])
    gy_proj = g * np.sin(data['phi']) * np.cos(data['theta'])
    gz_proj = g * np.cos(data['phi']) * np.cos(data['theta'])
    
    # 3. 比力 (Specific Force) = 运动加速度 - 重力
    # 另外加入 Bias (零偏) 测试去偏置算法
    bias_ax = 0.1
    bias_az = -0.2
    
    data['ax'] = (ax_kin - gx_proj) + bias_ax
    data['ay'] = (ay_kin - gy_proj)
    data['az'] = (az_kin - gz_proj) + bias_az # 包含重力反力，约等于 -9.8
    
    # 4. 注入高斯噪声 
    np.random.seed(999)
    for key in ['p', 'q', 'r']:
        data[key] += np.random.normal(0, 0.002, n) # 陀螺噪声
        
    for key in ['ax', 'ay', 'az']:
        data[key] += np.random.normal(0, 0.05, n)  # 加计噪声

    # ==============================
    # D. 气动数据
    # ==============================
    data['TAS'] = np.sqrt(data['u_g']**2 + data['v_g']**2 + data['w_g']**2)
    # 简单的代数关系填充
    data['alpha'] = np.arctan2(data['w_g'], data['u_g'])
    data['beta'] = np.arctan2(data['v_g'], data['TAS'])

    # ==============================
    # 保存与可视化
    # ==============================
    df = pd.DataFrame(data)
    
    # 强制列排序符合文档
    cols = ['time', 'delta_e', 'delta_a', 'delta_r', 'delta_t', 'RPM',
            'u_g', 'v_g', 'w_g', 'p', 'q', 'r', 'lat', 'lon', 'h',
            'phi', 'theta', 'psi', 'ax', 'ay', 'az', 'TAS', 'alpha', 'beta']
    df = df[cols]
    
    df.to_csv(filename, index=False, float_format='%.6f')
    print(f"测试数据生成完毕: {filename}")
    
    return df

if __name__ == "__main__":
    df = generate_full_test_data()
    
    # 画图验证 "Psi 跳变" 是否存在
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(df['time'], np.degrees(df['psi']), color='purple', label='Psi (deg)')
    plt.title('Check for Phase Wrap (+180 to -180 jump)')
    plt.grid(True)
    plt.ylabel('Heading (deg)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(df['time'], df['az'], color='green', label='Az (m/s^2)')
    plt.title('Az with Noise and Gravity')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()