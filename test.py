import torch
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from datetime import datetime
from torchdiffeq import odeint
from NeuralODEFunc import CoefficientNet, AerialSystemODE
from flight_scaler import FlightDataScaler
from train import FlightDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

def validate_sample_variation(sample, scaler, state_keys):
    """
    验证样本是否满足变化要求：每组物理量中至少2/3有明显变化
    在真实物理空间进行判断，每组物理量单独设置阈值

    Args:
        sample: 数据集样本字典
        scaler: FlightDataScaler 实例
        state_keys: 状态量键列表

    Returns:
        bool: 是否满足要求
    """
    gt_states_norm = sample['gt_states_norm']  # (T, 12)
    gt_force_real = sample['gt_force']  # (T, 6) 已经是真实物理空间

    # 反归一化状态量到真实物理空间
    gt_states_real = scaler.inverse_transform_vector(gt_states_norm, state_keys).numpy()

    # 定义物理量分组 (索引) 及其阈值
    groups = {
        'linear_vel': {
            'indices': [0, 1, 2],      # u, v, w (m/s)
            'threshold': 0.5,          # 线速度变化阈值
            'data': gt_states_real
        },
        'angular_vel': {
            'indices': [3, 4, 5],      # p, q, r (rad/s)
            'threshold': 0.1,          # 角速度变化阈值
            'data': gt_states_real
        },
        'attitude': {
            'indices': [6, 7, 8],      # phi, theta, psi (rad)
            'threshold': 0.05,         # 姿态角变化阈值
            'data': gt_states_real
        },
        'linear_accel': {
            'indices': [0, 1, 2],      # ax, ay, az (m/s^2)
            'threshold': 1.0,          # 线加速度变化阈值
            'data': gt_force_real.numpy()
        }
    }

    for group_name, group_info in groups.items():
        valid_count = 0
        for idx in group_info['indices']:
            data = group_info['data'][:, idx]
            variation = np.max(data) - np.min(data)  # 极差
            if variation > group_info['threshold']:
                valid_count += 1

        # 至少2/3有明显变化
        if valid_count < 2:
            return False

    return True

def main():
    # 1. 配置参数
    CONFIG = {
        'paths': {
            'scaler': 'scaler52.pkl',             
            'dataset': 'dataset52.pt',            
            'model_save': 'model_weights.pth'   
        },
        'data': {
            'window_size': 100,
            'dt': 0.02,
        },
        'props': {
            'm': 420,                   
            'S': 18.825,                
            'b': 9.804,                 
            'c': 1.932,                 
            'h0': 1100.5,       
            'T_offset': [0, 0, -0.75],  
            'I': [                      
                [ 539.246,        0, -105.374],
                [       0, 694.0101,        0],
                [-105.374,        0, 1018.916]
            ]
        }
    }

    # 2. 加载预处理数据、归一化器和 Dataset
    if not os.path.exists(CONFIG['paths']['model_save']):
        raise FileNotFoundError("找不到模型权重文件，请确认训练是否成功保存了模型！")

    scaler = FlightDataScaler().load(CONFIG['paths']['scaler'])
    data_dict = torch.load(CONFIG['paths']['dataset'], weights_only=False)
    test_dataset = FlightDataset(data_dict)
    print(f"数据集已加载，共有 {len(test_dataset)} 个序列切片。")

    # 2.1 加载无噪声数据集（用于对比）
    clean_dataset_path = 'dataset51.pt'
    test_dataset_clean = None
    if os.path.exists(clean_dataset_path):
        data_dict_clean = torch.load(clean_dataset_path, weights_only=False)
        test_dataset_clean = FlightDataset(data_dict_clean)
        print(f"无噪声数据集已加载，用于对比显示。")
    else:
        print(f"警告: 未找到无噪声数据集 {clean_dataset_path}，将不显示无噪声对比曲线。")

    # 3. 实例化模型并加载权重
    net = CoefficientNet().to(device)
    model = AerialSystemODE(
        neural_net=net, 
        known_props=CONFIG['props'], 
        scaler=scaler, 
        device=device
    ).to(device)
    
    model.load_state_dict(torch.load(CONFIG['paths']['model_save'], map_location=device))
    model.eval()
    print(f"模型权重已加载。")

    # 4. 智能抽取测试样本 (要求每组物理量中至少2/3有明显变化)
    state_keys = model.state_keys
    max_attempts = 500
    sample_idx = None
    for attempt in range(max_attempts):
        candidate = random.randint(10, len(test_dataset) - 10)
        if validate_sample_variation(test_dataset[candidate], scaler, state_keys):
            sample_idx = candidate
            break

    if sample_idx is None:
        sample_idx = random.randint(10, len(test_dataset) - 10)
        print(f"警告: 尝试 {max_attempts} 次后未找到满足变化要求的样本，使用随机样本。")

    # sample_idx = 359  # 手动指定测试样本
    print(f"本次抽取的测试切片 Index: {sample_idx} / {len(test_dataset)}")
    sample = test_dataset[sample_idx]

    x0 = sample['x0'].unsqueeze(0).to(device)
    controls = sample['controls'].unsqueeze(0).to(device)
    gt_states_norm = sample['gt_states_norm'].unsqueeze(0).to(device)
    gt_force_real = sample['gt_force'].unsqueeze(0).to(device)

    # 4.1 提取无噪声数据（如果可用）
    gt_states_norm_clean = None
    gt_force_real_clean = None
    if test_dataset_clean is not None:
        sample_clean = test_dataset_clean[sample_idx]
        gt_states_norm_clean = sample_clean['gt_states_norm'].unsqueeze(0).to(device)
        gt_force_real_clean = sample_clean['gt_force'].unsqueeze(0).to(device)

    T = CONFIG['data']['window_size']
    t_span = torch.linspace(0, (T - 1) * CONFIG['data']['dt'], T).to(device)

    with torch.no_grad():
        model.set_control_context(t_span, controls)
        pred_traj_norm = odeint(model, x0, t_span, method='rk4').permute(1, 0, 2)

        # 关键修改：使用预测轨迹计算力和力矩，而不是带噪声的真实状态
        # 这样可以避免噪声对力和力矩预测的影响
        force_dict = model.predict_forces_and_moments(pred_traj_norm, controls)
        pred_force_real = force_dict['pred_force'] # (1, T, 6)

    # 5. 反归一化为物理单位
    pred_flat = pred_traj_norm.squeeze(0)
    gt_flat = gt_states_norm.squeeze(0)

    pred_real = scaler.inverse_transform_vector(pred_flat, state_keys).cpu().numpy()
    gt_real = scaler.inverse_transform_vector(gt_flat, state_keys).cpu().numpy()
    time_axis = t_span.cpu().numpy()

    gt_force_np = gt_force_real.squeeze(0).cpu().numpy()
    pred_force_np = pred_force_real.squeeze(0).cpu().numpy()

    # 5.1 反归一化无噪声数据（如果可用）
    gt_real_clean = None
    gt_force_np_clean = None
    if gt_states_norm_clean is not None:
        gt_flat_clean = gt_states_norm_clean.squeeze(0)
        gt_real_clean = scaler.inverse_transform_vector(gt_flat_clean, state_keys).cpu().numpy()
        gt_force_np_clean = gt_force_real_clean.squeeze(0).cpu().numpy()

    # 6. 创建结果保存文件夹
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join('./results', current_time)
    os.makedirs(result_dir, exist_ok=True)

    # 7. 绘制加速度与角加速度的对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Aerodynamic Forces & Moments Prediction (#{sample_idx})', fontsize=16)

    force_keys = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)',
                  'dot_p (rad/s^2)', 'dot_q (rad/s^2)', 'dot_r (rad/s^2)']

    for i, key in enumerate(force_keys):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # 绘制带噪声的真实曲线
        ax.plot(time_axis, gt_force_np[:, i], label='Ground Truth (Noisy)', color='black', linewidth=2)

        # 绘制无噪声的真实曲线（如果可用）
        if gt_force_np_clean is not None:
            ax.plot(time_axis, gt_force_np_clean[:, i], label='Ground Truth (Clean)', color='blue', linestyle=':', linewidth=2)

        # 绘制预测曲线
        ax.plot(time_axis, pred_force_np[:, i], label='Prediction', color='red', linestyle='--', linewidth=2)

        ax.set_title(f'[{key}]', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend()

    # 同行共享比例尺 (相同的 y 轴范围，但中心可以不同)
    for row in range(2):
        # 计算该行每个子图的数据范围
        ranges = []
        centers = []
        for col in range(3):
            y_min_auto, y_max_auto = axes[row, col].get_ylim()
            data_range = y_max_auto - y_min_auto
            data_center = (y_max_auto + y_min_auto) / 2
            ranges.append(data_range)
            centers.append(data_center)

        # 使用最大范围作为统一比例尺
        max_range = max(ranges)
        max_range *= 1.1  # 10% padding

        # 为每个子图设置相同范围但不同中心
        for col in range(3):
            axes[row, col].set_ylim(centers[col] - max_range/2, centers[col] + max_range/2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(result_dir, 'force_prediction.png')
    plt.savefig(save_path, dpi=200)
    print(f"加速度与角加速度预测图已保存为: {save_path}")

    # 8. 绘制速度与角速度对比图 (u, v, w, p, q, r)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle(f'Velocities & Angular Velocities Trajectory Prediction (#{sample_idx})', fontsize=16)

    vel_keys = ['u (m/s)', 'v (m/s)', 'w (m/s)',
                'p (rad/s)', 'q (rad/s)', 'r (rad/s)']

    for i, key in enumerate(vel_keys):
        row = i // 3
        col = i % 3
        ax = axes2[row, col]

        # 前 6 个状态对应索引 0~5
        # 绘制带噪声的真实曲线
        ax.plot(time_axis, gt_real[:, i], label='Ground Truth (Noisy)', color='black', linewidth=2)

        # 绘制无噪声的真实曲线（如果可用）
        if gt_real_clean is not None:
            ax.plot(time_axis, gt_real_clean[:, i], label='Ground Truth (Clean)', color='blue', linestyle=':', linewidth=2)

        # 绘制预测曲线
        ax.plot(time_axis, pred_real[:, i], label='Prediction (Integration)', color='red', linestyle='--', linewidth=2)

        ax.set_title(f'[{key}]', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend()

    # 同行共享比例尺
    for row in range(2):
        ranges = []
        centers = []
        for col in range(3):
            y_min_auto, y_max_auto = axes2[row, col].get_ylim()
            data_range = y_max_auto - y_min_auto
            data_center = (y_max_auto + y_min_auto) / 2
            ranges.append(data_range)
            centers.append(data_center)

        max_range = max(ranges)
        max_range *= 1.1

        for col in range(3):
            axes2[row, col].set_ylim(centers[col] - max_range/2, centers[col] + max_range/2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path2 = os.path.join(result_dir, 'velocity_prediction.png')
    plt.savefig(save_path2, dpi=200)
    print(f"速度与角速度轨迹图已保存为: {save_path2}")

    # 9. 绘制位置与姿态角轨迹对比图 (x, y, z, phi, theta, psi)
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
    fig3.suptitle(f'Position & Attitude Trajectory Prediction (#{sample_idx})', fontsize=16)

    pos_keys = ['x (m)', 'y (m)', 'z (m)',
                'phi (rad)', 'theta (rad)', 'psi (rad)']
    state_indices = [9, 10, 11, 6, 7, 8]

    for i, (key, state_idx) in enumerate(zip(pos_keys, state_indices)):
        row = i // 3
        col = i % 3
        ax = axes3[row, col]

        # 绘制带噪声的真实曲线
        ax.plot(time_axis, gt_real[:, state_idx], label='Ground Truth (Noisy)', color='black', linewidth=2)

        # 绘制无噪声的真实曲线（如果可用）
        if gt_real_clean is not None:
            ax.plot(time_axis, gt_real_clean[:, state_idx], label='Ground Truth (Clean)', color='blue', linestyle=':', linewidth=2)

        # 绘制预测曲线
        ax.plot(time_axis, pred_real[:, state_idx], label='Prediction (Integration)', color='red', linestyle='--', linewidth=2)

        ax.set_title(f'[{key}]', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend()

    # 同行共享比例尺 (相同的 y 轴范围，但中心可以不同)
    for row in range(2):
        ranges = []
        centers = []
        for col in range(3):
            y_min_auto, y_max_auto = axes3[row, col].get_ylim()
            data_range = y_max_auto - y_min_auto
            data_center = (y_max_auto + y_min_auto) / 2
            ranges.append(data_range)
            centers.append(data_center)

        max_range = max(ranges)
        max_range *= 1.1

        for col in range(3):
            axes3[row, col].set_ylim(centers[col] - max_range/2, centers[col] + max_range/2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path3 = os.path.join(result_dir, 'trajectory_prediction.png')
    plt.savefig(save_path3, dpi=200)
    print(f"位置与姿态轨迹图已保存为: {save_path3}")

if __name__ == '__main__':
    main()