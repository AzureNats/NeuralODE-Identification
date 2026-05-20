import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint
from NeuralODEFunc import CoefficientNet, AerialSystemODE
from flight_scaler import FlightDataScaler
from data_processing import FlightDataPreprocessor
from relobralo import ReLoBRaLo
from torch.amp import autocast, GradScaler
import os
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# 一. 数据集包装
class FlightDataset(Dataset):
    def __init__(self, data_dict):
        """
        将 data_processing.py 输出的 Numpy 字典转换为 PyTorch Dataset。
        
        功能:
        1. 接收 Numpy 数据。
        2. 转换为 FloatTensor (保留在 CPU 防止显存爆炸)。
        3. 提供 __getitem__ 接口供 DataLoader 调用。
        
        Args:
            data_dict (dict): 包含 'input_ode', 'input_nn', 'labels' 的字典
        """
        super().__init__()
        # 1. ODE 初值 x0 (N, 12)
        self.x0 = torch.from_numpy(data_dict['input_ode']).float()
        
        # 2. NN 控制序列 (N, T, 4)
        self.controls = torch.from_numpy(data_dict['input_nn']).float()
        
        # 3. 物理真值 (labels)
        self.label_traj = torch.from_numpy(data_dict['labels']['traj']).float()   # (N, T, 12)
        self.label_force = torch.from_numpy(data_dict['labels']['force']).float() # (N, T, 6)

        # 4. 归一化状态序列 (用于监督)
        self.states_norm = torch.from_numpy(data_dict['states_norm']).float()
        
        # 获取样本数量
        self.n_samples = self.x0.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        返回单个样本。
        注意: 这里返回的 Tensor 依然在 CPU 上，会在 Training Loop 中被搬运到 GPU。
        """
        return {
            'x0':             self.x0[idx],           # 归一化初值
            'controls':       self.controls[idx],     # 归一化控制序列
            'gt_traj':        self.label_traj[idx],   # 物理轨迹真值
            'gt_force':       self.label_force[idx],  # 物理力/导数真值
            'gt_states_norm': self.states_norm[idx]   # 归一化状态序列
        }

# 二. 混合损失函数
class HybridLoss(nn.Module):
    def __init__(self):
        """
        混合损失函数，返回未加权的各项 Loss。
        轨迹损失拆分为速度损失和运动学损失；
        力损失拆分为纵向和横航向，加权由外部 ReLoBRaLo 负责。
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.lon_idx = [0, 2, 4]  # ax, az, dot_q
        self.lat_idx = [1, 3, 5]  # ay, dot_p, dot_r

    def forward(self, pred_traj_norm, gt_traj_norm, pred_force_real, gt_force_real):
        """
        Returns:
            loss_vel (Tensor): 速度+角速度 MSE (归一化空间)
            loss_kin (Tensor): 姿态+位置 MSE (归一化空间)
            loss_force_lon (Tensor): 纵向力误差 [ax, az, dot_q] (方差归一化)
            loss_force_lat (Tensor): 横航向力误差 [ay, dot_p, dot_r] (方差归一化)
        """
        loss_vel = self.mse(pred_traj_norm[..., :6], gt_traj_norm[..., :6])
        loss_kin = self.mse(pred_traj_norm[..., 6:], gt_traj_norm[..., 6:])

        mse_per_dim = torch.mean((pred_force_real - gt_force_real)**2, dim=(0, 1))
        gt_var = torch.var(gt_force_real, dim=(0, 1), unbiased=False)
        gt_var = torch.clamp(gt_var, min=1e-3)
        norm_mse = mse_per_dim / gt_var

        loss_force_lon = torch.mean(norm_mse[self.lon_idx])
        loss_force_lat = torch.mean(norm_mse[self.lat_idx])

        return loss_vel, loss_kin, loss_force_lon, loss_force_lat
    
    
# 三. 主函数
def main():
    start_time = time.time()

    # 1. 全局配置
    CONFIG = {
        # 路径配置
        'paths': {
            'raw_csv': 'Document52.csv',         # 原始飞行数据 (CSV格式)
            'scaler': 'scaler52.pkl',            # 归一化参数保存路径 (Pickle)
            'dataset': 'dataset52.pt',           # 预处理后的数据集保存路径 (PyTorch Tensor)
            'pre_wei': 'pretrained_coeffs.pth',  # 预训练模型权重路径 (PyTorch Model)
            'model_save': 'model_weights.pth'    # 模型权重保存路径 (PyTorch Model)
        },
        
        # 数据预处理参数
        'preprocess': {
            'is_windless': True,                # 无风仿真标志 (True: 地速=空速, False: 需考虑风速)
            'lever_arm': [0, 0, 0],             # 杆臂效应修正向量 [x, y, z] (单位: m, None表示不修正)
        },
        
        # 数据集参数
        'data': {
            'window_size': 100,        # 时间序列切片长度 (积分帧数) 默认100帧(2s @ 50Hz)
            'stride': 20,              # 滑动窗口步长 (越小切片越多, 但重叠越高)
            'dt': 0.02,                # 采样时间间隔 (单位: s, 对应50Hz)
        },
        
        # 训练超参数
        'train': {
            'batch_size': 256,         # 批大小
            'max_lr': 1e-3,            # 初始学习率
            'min_lr': 1e-5,            # 最小学习率
            'epochs': 250,             # 总训练轮数
            'save_interval': 10,       # 每隔多少个Epoch保存一次模型
            'num_workers': 4,          # DataLoader工作线程数
        },

        # ReLoBRaLo 及正则化参数
        'loss': {
            'alpha': 0.95,             # 指数衰减率 (控制"记住过去"的能力, 推荐: 0.9-0.99)
            'rho': 0.999,              # Saudade 伯努利期望值 (控制回溯频率, 推荐: 0.999)
            'temperature': 1.0,        # Softmax 温度 (越小分布越尖锐, 推荐: 0.5-2.0)
            'base_weights': [1.0, 1.0, 1.0, 1.0],  # [vel, kin, force_lon, force_lat] 基础缩放因子
            'enable_jacobian': True,   # Jacobian 正则化开关
            'enable_hessian': True,    # Hessian 正则化开关
            'w_jac': 10.0,             # Jacobian 正则化固定权重
            'w_hes': 2.0,              # Hessian 正则化固定权重
        },
        
        # 物理参数
        'props': {
            'm': 420,                   # 质量 (kg)
            'S': 18.825,                # 机翼参考面积 (m^2)
            'b': 9.804,                 # 翼展 (m)
            'c': 1.932,                 # 平均气动弦长 (m)
            'h0': 1100.5,               # 起飞高度 (m)
            'T_offset': [0, 0, -0.75],  # 推力线偏心距 (m)
            'I': [                      # 惯量 (kg*m^2)
                [ 539.246,      0.0, -105.374],
                [     0.0, 694.0101,      0.0],
                [-105.374,      0.0, 1018.916]
            ]
        }
    }
    
    print(f"正在初始化训练，使用设备: {device}")

    # 2. 数据准备
    # 加载 scaler 和 dataset
    paths = CONFIG['paths']
    scaler = FlightDataScaler()
    data_dict = None

    has_cache = os.path.exists(paths['scaler']) and os.path.exists(paths['dataset'])
    if has_cache:
        print("正在直接加载本地缓存文件...")
        scaler.load(paths['scaler'])
        data_dict = torch.load(paths['dataset'], weights_only=False)
        print("缓存加载成功！")
    else:
        print("正在运行预处理流水线...")
        if not os.path.exists(paths['raw_csv']):
            raise FileNotFoundError(f"找不到原始数据文件: {paths['raw_csv']}")

        pipeline = FlightDataPreprocessor()
        data_dict = pipeline.run_pipeline(CONFIG)
        scaler.load(paths['scaler'])
    
    # 实例化 Dataset 和 DataLoader
    train_dataset = FlightDataset(data_dict)
    train_loader = DataLoader(
        train_dataset,
        batch_size = CONFIG['train']['batch_size'],
        shuffle = True,
        num_workers = CONFIG['train']['num_workers'],
        pin_memory = True,
        persistent_workers = True
    )

    # 初始化用于记录历史 loss 的列表
    history_total = []
    history_vel = []
    history_kin = []
    history_force_lon = []
    history_force_lat = []
    history_jac = []
    history_hes = []
    history_w_vel = []
    history_w_kin = []
    history_w_force_lon = []
    history_w_force_lat = []
    
    print(f"数据加载完成。样本数: {len(train_dataset)}")

    # 3. 实例化模型
    net = CoefficientNet().to(device)
    net.load_state_dict(torch.load(CONFIG['paths']['pre_wei']))
    model = AerialSystemODE(
        neural_net=net, 
        known_props=CONFIG['props'], 
        scaler=scaler, 
        device=device
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['train']['max_lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max = CONFIG['train']['epochs'], 
        eta_min = CONFIG['train']['min_lr']
    )
    loss_fn = HybridLoss().to(device)
    relo = ReLoBRaLo(
        m=4,
        alpha=CONFIG['loss']['alpha'],
        rho=CONFIG['loss']['rho'],
        temperature=CONFIG['loss']['temperature'],
        base_weights=CONFIG['loss']['base_weights'],
        max_epochs=CONFIG['train']['epochs']
    )

    # 4. 准备积分时间向量
    # T = window_size, dt = 0.02
    # shape: (T,) -> [0.00, 0.02, ..., 1.98]
    t_span = torch.linspace(
        0,
        (CONFIG['data']['window_size'] - 1) * CONFIG['data']['dt'],
        CONFIG['data']['window_size']
    ).to(device)

    # 5. 训练主循环
    print("开始训练...")
    amp_scaler = GradScaler('cuda')

    for epoch in range(CONFIG['train']['epochs']):
        model.train()

        epoch_loss_total = 0.0
        epoch_loss_vel = 0.0
        epoch_loss_kin = 0.0
        epoch_loss_force_lon = 0.0
        epoch_loss_force_lat = 0.0
        epoch_loss_jac = 0.0
        epoch_loss_hes = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # A. 搬运数据到 GPU
            x0 = batch['x0'].to(device, non_blocking=True)                         # (B, 12)
            controls = batch['controls'].to(device, non_blocking=True)             # (B, T, 4)
            gt_traj = batch['gt_traj'].to(device, non_blocking=True)               # (B, T, 12)
            gt_force = batch['gt_force'].to(device, non_blocking=True)             # (B, T, 6)
            gt_states_norm = batch['gt_states_norm'].to(device, non_blocking=True) # (B, T, 12)

            optimizer.zero_grad()

            with autocast('cuda'):
                # B. 注入控制上下文
                model.set_control_context(t_span, controls)

                # C. 积分 -> 得到轨迹
                pred_traj_norm = odeint(model, x0, t_span, method='rk4')
                pred_traj_norm = pred_traj_norm.permute(1, 0, 2)

                # D. 诊断 -> 得到力
                force_dict = model.predict_forces_and_moments(gt_states_norm, controls)
                pred_force_real = force_dict['pred_force'] # (B, T, 6)

                # E. 计算 Loss
                loss_vel, loss_kin, loss_force_lon, loss_force_lat = loss_fn(pred_traj_norm, gt_states_norm, pred_force_real, gt_force)
                w_vel, w_kin, w_force_lon, w_force_lat = relo.get_weights()

            # Jacobian/Hessian 正则化需要 float32 精度，放在 autocast 外
            use_jac = CONFIG['loss']['enable_jacobian']
            use_hes = CONFIG['loss']['enable_hessian']

            loss_jac = model.net.compute_jacobian_regularization(gt_states_norm.float(), controls.float()) if use_jac else torch.tensor(0.0, device=device)
            loss_hes = model.net.compute_hessian_regularization(gt_states_norm.float(), controls.float()) if use_hes else torch.tensor(0.0, device=device)

            loss = (w_vel*loss_vel + w_kin*loss_kin
                    + w_force_lon*loss_force_lon + w_force_lat*loss_force_lat
                    + (CONFIG['loss']['w_jac']*loss_jac if use_jac else 0.0)
                    + (CONFIG['loss']['w_hes']*loss_hes if use_hes else 0.0))

            log_dict = {
                'loss_vel': loss_vel.item(),
                'loss_kin': loss_kin.item(),
                'loss_force_lon': loss_force_lon.item(),
                'loss_force_lat': loss_force_lat.item(),
                'loss_jac': loss_jac.item(),
                'loss_hes': loss_hes.item(),
            }

            # F. 反向传播 (AMP)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            epoch_loss_total += loss.item()
            epoch_loss_vel += log_dict['loss_vel']
            epoch_loss_kin += log_dict['loss_kin']
            epoch_loss_force_lon += log_dict['loss_force_lon']
            epoch_loss_force_lat += log_dict['loss_force_lat']
            epoch_loss_jac += log_dict['loss_jac']
            epoch_loss_hes += log_dict['loss_hes']

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.6f} "
                      f"(Vel: {log_dict['loss_vel']:.6f}, "
                      f"Kin: {log_dict['loss_kin']:.6f}, "
                      f"F_lon: {log_dict['loss_force_lon']:.6f}, "
                      f"F_lat: {log_dict['loss_force_lat']:.6f}, "
                      f"Jac: {log_dict['loss_jac']:.6f}, "
                      f"Hes: {log_dict['loss_hes']:.6f}) "
                      f"[w: V={w_vel:.3f} K={w_kin:.3f} Flon={w_force_lon:.3f} Flat={w_force_lat:.3f}]")

        avg_total = epoch_loss_total / len(train_loader)
        avg_vel = epoch_loss_vel / len(train_loader)
        avg_kin = epoch_loss_kin / len(train_loader)
        avg_force_lon = epoch_loss_force_lon / len(train_loader)
        avg_force_lat = epoch_loss_force_lat / len(train_loader)
        avg_jac = epoch_loss_jac / len(train_loader)
        avg_hes = epoch_loss_hes / len(train_loader)

        history_total.append(avg_total)
        history_vel.append(avg_vel)
        history_kin.append(avg_kin)
        history_force_lon.append(avg_force_lon)
        history_force_lat.append(avg_force_lat)
        history_jac.append(avg_jac)
        history_hes.append(avg_hes)

        # ReLoBRaLo 权重更新
        relo.update([avg_vel, avg_kin, avg_force_lon, avg_force_lat])
        w_v, w_k, w_fl, w_fla = relo.get_weights()
        history_w_vel.append(w_v)
        history_w_kin.append(w_k)
        history_w_force_lon.append(w_fl)
        history_w_force_lat.append(w_fla)

        print(f"Epoch {epoch} 完成 | 平均 Loss: {avg_total:.6f} | "
              f"下轮权重: V={w_v:.3f} K={w_k:.3f} Flon={w_fl:.3f} Flat={w_fla:.3f}")
        
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), CONFIG['paths']['model_save'])
            print(f"模型已保存至 {CONFIG['paths']['model_save']}")

    # 6. 绘制训练曲线 (双子图: Loss + ReLoBRaLo 数据权重)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                    gridspec_kw={'height_ratios': [2, 1]})
    epochs_range = range(1, CONFIG['train']['epochs'] + 1)

    # 上图: Loss 曲线
    use_jac = CONFIG['loss']['enable_jacobian']
    use_hes = CONFIG['loss']['enable_hessian']

    ax1.plot(epochs_range, history_total, label='Total Loss', color='black', linewidth=2)
    ax1.plot(epochs_range, history_vel, label='Velocity Loss', color='blue', linestyle='--')
    ax1.plot(epochs_range, history_kin, label='Kinematic Loss', color='cyan', linestyle='--')
    ax1.plot(epochs_range, history_force_lon, label='Force Loss (Lon)', color='red', linestyle='-.')
    ax1.plot(epochs_range, history_force_lat, label='Force Loss (Lat)', color='orange', linestyle='-.')
    ax1.plot(epochs_range, history_jac, label=f'Jacobian Loss{"" if use_jac else " (OFF)"}', color='green', linestyle=':')
    ax1.plot(epochs_range, history_hes, label=f'Hessian Loss{"" if use_hes else " (OFF)"}', color='purple', linestyle=':')
    ax1.set_ylabel('Loss Value')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_title('Neural ODE Training Convergence')

    # 下图: ReLoBRaLo 动态权重
    ax2.plot(epochs_range, history_w_vel, label='w_vel', color='blue', linewidth=1.5)
    ax2.plot(epochs_range, history_w_kin, label='w_kin', color='cyan', linewidth=1.5)
    ax2.plot(epochs_range, history_w_force_lon, label='w_force_lon', color='red', linewidth=1.5)
    ax2.plot(epochs_range, history_w_force_lat, label='w_force_lat', color='orange', linewidth=1.5)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Weight')
    ax2.legend()
    ax2.grid(True, ls="--", alpha=0.5)

    # 动态生成标题，显示实际启用的正则化项
    reg_status = []
    if use_jac:
        reg_status.append(f'w_jac={CONFIG["loss"]["w_jac"]}')
    else:
        reg_status.append('w_jac=OFF')
    if use_hes:
        reg_status.append(f'w_hes={CONFIG["loss"]["w_hes"]}')
    else:
        reg_status.append('w_hes=OFF')
    ax2.set_title(f'ReLoBRaLo Data Weights (Reg fixed: {", ".join(reg_status)})')

    plt.tight_layout()
    save_fig_path = 'training_loss_curve.png'
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"Loss 曲线图已保存至: {save_fig_path}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"训练程序运行结束, 总耗时 {int(hours):02d}小时 {int(minutes):02d}分钟 {int(seconds):02d}秒")

if __name__ == '__main__':
    main()