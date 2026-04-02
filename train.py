import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint
from NeuralODEFunc import CoefficientNet, AerialSystemODE
from flight_scaler import FlightDataScaler
from data_processing import FlightDataPreprocessor
import os
import matplotlib.pyplot as plt

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
    def __init__(self, weight_traj, weight_force):
        """
        定义混合损失函数: L_total = w1 * L_traj + w2 * L_force
        
        Args:
            weight_traj (float): 轨迹误差的权重
            weight_force (float): 动力学/力误差的权重
        """
        super().__init__()
        self.w_traj = weight_traj
        self.w_force = weight_force
        self.mse = nn.MSELoss()

    def forward(self, pred_traj_norm, gt_traj_norm, pred_force_real, gt_force_real):
        """
        计算总 Loss。
        
        关键逻辑:
        - 轨迹 Loss: 归一化空间计算。
        - 力 Loss: 物理空间计算，并使用动态方差归一化。
        
        Args:
            pred_traj_norm (Tensor): ODE 积分出的归一化轨迹 (B, T, 12)
            gt_traj_norm (Tensor): 真实的归一化轨迹 (B, T, 12)
            pred_force_real (Tensor): Teacher Forcing 预测的物理力 (B, T, 6)
            gt_force_real (Tensor): IMU 测量的物理力/导数 (B, T, 6)
            scaler (FlightDataScaler): 用于反归一化的 scaler 实例
            
        Returns:
            loss_total (Tensor): 总 Loss (标量)
            log_dict (dict): 分项 Loss (用于日志打印)
        """
        # 1. 计算轨迹损失 (L_trajectory)
        loss_traj = self.mse(pred_traj_norm, gt_traj_norm)

        # 2. 计算动力学损失 (L_force)
        # 计算 6 个物理维度各自的 MSE 误差 -> 形状: (6,)
        # dim=(0, 1) 表示在 Batch(B) 和 Time(T) 维度上求均值
        mse_per_dim = torch.mean((pred_force_real - gt_force_real)**2, dim=(0, 1))
        
        # 计算 6 个物理维度各自的真实方差 -> 形状: (6,)
        gt_var = torch.var(gt_force_real, dim=(0, 1), unbiased=False)
        gt_var = torch.clamp(gt_var, min=1e-3)
        
        loss_force = torch.mean(mse_per_dim / gt_var)

        # 3. 加权求和
        loss_total = self.w_traj * loss_traj + self.w_force * loss_force
        log_dict = {
            'loss_traj': loss_traj.item(),
            'loss_force': loss_force.item()
        }
        
        return loss_total, log_dict
    
    
# 三. 主函数
def main():
    # 1. 全局配置
    CONFIG = {
        # 路径配置
        'paths': {
            'raw_csv': 'Document41.csv',          # 原始飞行数据 (CSV格式)
            'scaler': 'scaler41.pkl',             # 归一化参数保存路径 (Pickle)
            'dataset': 'dataset41.pt',            # 预处理后的数据集保存路径 (PyTorch Tensor)
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
            'batch_size': 64,          # 批大小
            'learning_rate': 1e-5,     # 初始学习率
            'min_lr':1e-7,             # 最小学习率
            'epochs': 150,             # 总训练轮数
            'save_interval': 10,       # 每隔多少个Epoch保存一次模型
            'num_workers': 0,          # DataLoader工作线程数 (Windows建议设为0)
        },

        # 损失函数权重
        'loss': {
            'w_traj': 1.0,             # 轨迹积分误差的权重 (L_traj)
            'w_force': 1.0,            # 动力学/力误差的权重 (L_force)
            'w_jac': 100.0,            # 雅可比正则化权重 (L_jac)
        },
        
        # 物理参数
        'props': {
            'm': 420,                   # 质量 (kg)
            'S': 18.825,                # 机翼参考面积 (m^2)
            'b': 9.804,                 # 翼展 (m)
            'c': 1.932,                 # 平均气动弦长 (m)
            'base_altitude': 0.0,       # 起飞高度 (m)
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
        num_workers = CONFIG['train']['num_workers']
    )

    # 初始化用于记录历史 loss 的列表
    history_total = []
    history_traj = []
    history_force = []
    history_jac = []
    
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
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['train']['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max = CONFIG['train']['epochs'], 
        eta_min = CONFIG['train']['min_lr']
    )
    loss_fn = HybridLoss(
        weight_traj = CONFIG['loss']['w_traj'],
        weight_force = CONFIG['loss']['w_force']
    ).to(device)

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
    # 定义预热参数
    warmup_start_epoch = 10  # 从第几轮开始引入轨迹 Loss
    warmup_epochs = 30       # 预热过渡期需要多少轮
    start_w_traj = 1e-2      # 轨迹 Loss 的初始极小权重
    end_w_traj = CONFIG['loss']['w_traj'] # 最终目标权重

    for epoch in range(CONFIG['train']['epochs']):
        model.train()

        epoch_loss_total = 0.0
        epoch_loss_traj = 0.0
        epoch_loss_force = 0.0
        epoch_loss_jac = 0.0

        # 动态调整 Loss 权重 (指数预热策略)
        if epoch < warmup_start_epoch:
            # 前 10 轮：关闭轨迹积分 Loss，仅拟合瞬间的气动力
            loss_fn.w_traj = 0.0
            loss_fn.w_force = 1.0
        else:
            # 10 轮之后：开启轨迹积分 Loss，指数平滑过渡
            current_step = min(epoch - warmup_start_epoch, warmup_epochs)
            current_w_traj = start_w_traj * ((end_w_traj / start_w_traj) ** (current_step / warmup_epochs))
            loss_fn.w_traj = current_w_traj
            loss_fn.w_force = CONFIG['loss']['w_force']

        for batch_idx, batch in enumerate(train_loader):
            # A. 搬运数据到 GPU
            x0 = batch['x0'].to(device)                         # (B, 12)
            controls = batch['controls'].to(device)             # (B, T, 4)
            gt_traj = batch['gt_traj'].to(device)               # (B, T, 12)
            gt_force = batch['gt_force'].to(device)             # (B, T, 6)
            gt_states_norm = batch['gt_states_norm'].to(device) # (B, T, 12)

            optimizer.zero_grad()
            
            # B. 注入控制上下文 (Context Injection)
            # 告诉模型：在接下来积分的这段时间里，舵面是怎么动的
            model.set_control_context(t_span, controls)
            
            # C. 积分 (Forward - Integration) -> 得到轨迹
            # odeint 返回形状是 (T, B, 12)，需要转置为 (B, T, 12) 以匹配 Dataset
            # method='rk4' 显存占用适中，'dopri5' 精度更高但更慢
            pred_traj_norm = odeint(model, x0, t_span, method='rk4') 
            pred_traj_norm = pred_traj_norm.permute(1, 0, 2)
            
            # D. 诊断 (Diagnostic - Teacher Forcing) -> 得到力
            force_dict = model.predict_forces_and_moments(gt_states_norm, controls)
            
            # 提取预测的加速度 (用于和 IMU 数据 gt_force 对比)
            pred_force_real = force_dict['pred_force'] # (B, T, 6)
            
            # E. 计算 Loss
            loss, log_dict = loss_fn(pred_traj_norm, gt_states_norm, pred_force_real, gt_force)
            loss_jac = model.net.compute_jacobian_regularization(gt_states_norm, controls)
            loss += loss_jac * CONFIG['loss']['w_jac']
            log_dict['loss_jac'] = loss_jac.item()
            
            # F. 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss_total += loss.item()
            epoch_loss_traj += log_dict['loss_traj']
            epoch_loss_force += log_dict['loss_force']
            epoch_loss_jac += log_dict['loss_jac']
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.6f} "
                      f"(Traj: {log_dict['loss_traj']:.6f}, "
                      f"Force: {log_dict['loss_force']:.6f}, "
                      f"Jac: {log_dict['loss_jac']:.6f})")

        avg_total = epoch_loss_total / len(train_loader)
        avg_traj = epoch_loss_traj / len(train_loader)
        avg_force = epoch_loss_force / len(train_loader)
        avg_jac = epoch_loss_jac / len(train_loader)

        history_total.append(avg_total)
        history_traj.append(avg_traj)
        history_force.append(avg_force)
        history_jac.append(avg_jac)

        print(f"Epoch {epoch} 完成 | 平均 Loss: {avg_total:.6f}")
        
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), CONFIG['paths']['model_save'])
            print(f"模型已保存至 {CONFIG['paths']['model_save']}")

    # 6. 绘制训练曲线
    plt.figure(figsize=(16, 9))
    epochs_range = range(1, CONFIG['train']['epochs'] + 1)
    
    plt.plot(epochs_range, history_total, label='Total Loss', color='black', linewidth=2)
    plt.plot(epochs_range, history_traj, label='Trajectory Loss (Raw MSE)', color='blue', linestyle='--')
    plt.plot(epochs_range, history_force, label='Force Loss (Var-Normalized)', color='red', linestyle='-.')
    plt.plot(epochs_range, history_jac, label='Jacobian Loss (Frobenius Norm)', color='green', linestyle=':')
    
    plt.title('Neural ODE Training Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    
    plt.yscale('log') 
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    save_fig_path = 'training_loss_curve.png'
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"Loss 曲线图已保存至: {save_fig_path}")
    plt.show()

if __name__ == '__main__':
    main()