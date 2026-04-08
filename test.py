import torch
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
from torchdiffeq import odeint
from NeuralODEFunc import CoefficientNet, AerialSystemODE
from flight_scaler import FlightDataScaler
from train import FlightDataset 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

def main():
    # 1. 配置参数
    CONFIG = {
        'paths': {
            'scaler': 'scaler41.pkl',             
            'dataset': 'dataset41.pt',            
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

    # 4. 抽取测试样本并进行预测
    sample_idx = random.randint(10, len(test_dataset) - 10)
    sample_idx = 714
    print(f"本次随机抽取的测试切片 Index: {sample_idx} / {len(test_dataset)}")
    sample = test_dataset[sample_idx]
    
    x0 = sample['x0'].unsqueeze(0).to(device)                         
    controls = sample['controls'].unsqueeze(0).to(device)             
    gt_states_norm = sample['gt_states_norm'].unsqueeze(0).to(device) 
    gt_force_real = sample['gt_force'].unsqueeze(0).to(device)

    T = CONFIG['data']['window_size']
    t_span = torch.linspace(0, (T - 1) * CONFIG['data']['dt'], T).to(device)

    with torch.no_grad(): 
        model.set_control_context(t_span, controls)
        pred_traj_norm = odeint(model, x0, t_span, method='rk4').permute(1, 0, 2)
        force_dict = model.predict_forces_and_moments(gt_states_norm, controls)
        pred_force_real = force_dict['pred_force'] # (1, T, 6)

    # 5. 反归一化为物理单位
    state_keys = model.state_keys
    
    pred_flat = pred_traj_norm.squeeze(0)
    gt_flat = gt_states_norm.squeeze(0)
    
    pred_real = scaler.inverse_transform_vector(pred_flat, state_keys).cpu().numpy()
    gt_real = scaler.inverse_transform_vector(gt_flat, state_keys).cpu().numpy()
    time_axis = t_span.cpu().numpy()

    gt_force_np = gt_force_real.squeeze(0).cpu().numpy()
    pred_force_np = pred_force_real.squeeze(0).cpu().numpy()

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
        
        ax.plot(time_axis, gt_force_np[:, i], label='Ground Truth', color='black', linewidth=2)
        ax.plot(time_axis, pred_force_np[:, i], label='Prediction', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f'[{key}]', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0: 
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    save_path = os.path.join(result_dir, 'force_prediction.png')
    plt.savefig(save_path, dpi=200)
    print(f"加速度与角加速度预测图已保存为: {save_path}")

    # 8. 绘制速度与角速度对比图 (u, v, w, p, q, r)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle(f'Velocities & Angular Velocities Trajectory (#{sample_idx})', fontsize=16)

    vel_keys = ['u (m/s)', 'v (m/s)', 'w (m/s)', 
                'p (rad/s)', 'q (rad/s)', 'r (rad/s)']

    for i, key in enumerate(vel_keys):
        row = i // 3
        col = i % 3
        ax = axes2[row, col]
        
        # 前 6 个状态对应索引 0~5
        ax.plot(time_axis, gt_real[:, i], label='Ground Truth', color='black', linewidth=2)
        ax.plot(time_axis, pred_real[:, i], label='Prediction (Integration)', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f'[{key}]', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0: 
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    save_path2 = os.path.join(result_dir, 'velocity_prediction.png')
    plt.savefig(save_path2, dpi=200)
    print(f"速度与角速度轨迹图已保存为: {save_path2}")

    # 9. 绘制姿态角与位置轨迹对比图 (phi, theta, psi, x, y, z)
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
    fig3.suptitle(f'Attitude & Position Trajectory (#{sample_idx})', fontsize=16)

    pos_keys = ['phi (rad)', 'theta (rad)', 'psi (rad)', 
                'x (m)', 'y (m)', 'z (m)']

    for i, key in enumerate(pos_keys):
        row = i // 3
        col = i % 3
        ax = axes3[row, col]
        
        # 后 6 个状态对应索引 6~11
        state_idx = i + 6
        ax.plot(time_axis, gt_real[:, state_idx], label='Ground Truth', color='black', linewidth=2)
        ax.plot(time_axis, pred_real[:, state_idx], label='Prediction (Integration)', color='red', linestyle='--', linewidth=2)
        
        ax.set_title(f'[{key}]', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0: 
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    save_path3 = os.path.join(result_dir, 'trajectory_prediction.png')
    plt.savefig(save_path3, dpi=200)
    print(f"姿态与位置轨迹图已保存为: {save_path3}")

if __name__ == '__main__':
    main()