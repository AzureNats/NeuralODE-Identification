import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import scipy.signal
import torch
from flight_scaler import FlightDataScaler
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class FlightDataPreprocessor:
    def __init__(self, t_sample=0.02):
        """
        初始化预处理器
        
        Args:
            t_sample (float): 数据采样时间间隔(s), 默认0.02s (50Hz)
        """
        self.dt = t_sample
        
        # 协议清单：标准变量名与含义说明
        self.variable_protocol = {
            # --- 状态量 ---
            'TAS':   '真空速 (m/s)',
            'Q':     '动压 (Pa)',
            'alpha': '攻角 (rad)',
            'beta':  '侧滑角 (rad)',
            
            # --- 地速 (NED系) ---
            'u_g':   '北向地速 (m/s)',
            'v_g':   '东向地速 (m/s)',
            'w_g':   '地向地速 (m/s, 下为正)',
            
            # --- 角速度 (机体轴) ---
            'p':     '滚转角速度 (rad/s)',
            'q':     '俯仰角速度 (rad/s)',
            'r':     '偏航角速度 (rad/s)',
            
            # --- 位置 (LLA) ---
            'lat':   '纬度 (deg)',
            'lon':   '经度 (deg)',
            'h':     '海拔高度 (m)',
            
            # --- 姿态 (欧拉角) ---
            'phi':   '滚转角 (rad)',
            'theta': '俯仰角 (rad)',
            'psi':   '偏航角 (rad)',
            
            # --- 加速度 (IMU, 机体轴) ---
            'ax':    '纵向加速度 (m/s^2)',
            'ay':    '横向加速度 (m/s^2)',
            'az':    '垂向加速度 (m/s^2)',
            
            # --- 控制量 ---
            'delta_e': '升降舵偏角 (%)',
            'delta_a': '副翼偏角 (%)',
            'delta_r': '方向舵偏角 (%)',
            'delta_t': '油门 (%)',
            'rpm' :    '转速 (rpm, 与油门二选一)'
        }

    def run_pipeline(self, config):
        """
        数据预处理主函数，执行完整的预处理流水线。

        Args:
            file_path (str): CSV文件路径
            save_scaler_path (str): 归一化参数保存路径
            save_dataset_path (str): 训练数据集保存路径
            is_windless (bool): 无风仿真标志
            lever_arm (list/array, optional): 杆臂效应修正向量
            window_size (int): 切片窗口长度
            stride (int): 滑动步长 
        
        Returns:
            dict: 包含训练所需数据集的字典
                - 'input_ode_x0': (N, 12) 归一化的初始状态
                - 'input_nn_controls': (N, T, 4) 归一化的控制量序列
                - 'labels': 包含物理真值的字典
                    - 'traj': (N, T, 12) 物理状态轨迹真值
                    - 'force': (N, T, 6) 物理力/导数真值
        """
        # 0. 参数解包
        file_path = config['paths']['raw_csv']
        save_scaler_path = config['paths']['scaler']
        save_dataset_path = config['paths']['dataset']
        
        is_windless = config['preprocess']['is_windless']
        lever_arm = config['preprocess']['lever_arm']
        
        window_size = config['data']['window_size']
        stride = config['data']['stride']
        self.dt = config['data']['dt']
        self.h0 = config['props']['h0']

        # 1. 加载与清洗
        df = self._load_and_check(file_path, is_windless)
        
        # 2. 经纬度转 NED
        df = self._convert_gps_to_ned(df)
        df_raw_backup = df.copy()
        
        # 3. 低通滤波
        # df = self._apply_low_pass_filter(df)

        # 4. 状态变量重构
        df = self._reconstruct_state_variables(df, is_windless)

        # 5. 数值微分与导数监督
        df = self._prepare_derivative_labels(df, lever_arm)

        # 6. 数据归一化
        df_norm = self._normalize_data(df, save_scaler_path)

        # 7. 可视化
        if config['preprocess'].get('visualize', False):
           self.visualize_comparison(df_raw_backup, df, save_dir='./viz_results')
        
        # 8. 轨迹切片
        dataset_dict = self._create_trajectory_slices(df, df_norm, window_size, stride)
        
        # 9. 保存处理好的数据集
        if save_dataset_path:
            print(f"正在保存处理后的数据集至: {save_dataset_path} ...")
            torch.save(dataset_dict, save_dataset_path)
            print("数据集保存完成。")

        print(f"预处理全部完成。生成切片数量: {len(dataset_dict['input_ode'])}")
        return dataset_dict

    def _load_and_check(self, file_path, is_windless):
        """
        读取CSV, 并检查完整性
        
        Args:
            file_path (str): CSV文件路径
            is_windless (bool): 是否为无风仿真环境 
        
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
            
        print(f"正在读取数据: {file_path} ...")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        missing_cols = []
        
        # 1. 检查动力数据 (二选一)
        if 'delta_t' not in df.columns and 'rpm' not in df.columns:
            missing_cols.append("delta_t 或 rpm")
            
        # 2. 检查气动数据 (有风模式)
        if not is_windless:
            for col in ['TAS', 'alpha', 'beta']:
                if col not in df.columns:
                    missing_cols.append(col)
        else:
            print("提示: 当前设置为无风仿真模式。")

        # 3. 检查其他所有基础变量
        check_list = [k for k in self.variable_protocol.keys() 
                      if k not in ['TAS', 'alpha', 'beta', 'Q', 'delta_t', 'rpm']]
        
        for col in check_list:
            if col not in df.columns:
                missing_cols.append(col)
                
        if missing_cols:
            raise ValueError(f"CSV文件缺失必要列: {missing_cols}")
        
        # 4. 检查数据丢包
        if df.isnull().any().any():
            nan_count = df.isnull().sum().sum()
            print(f"警告: 检测到 {nan_count} 个缺失值，正在尝试线性插值修复...")
            
            max_gap_time = 0.1
            limit_frames = int(max_gap_time / self.dt)
            df = df.interpolate(method='linear', limit=limit_frames, limit_direction='both')
            
            if df.isnull().any().any():
                cols_with_nan = df.columns[df.isnull().any()].tolist()
                raise ValueError(f"数据不可用，请重新采集。受影响列: {cols_with_nan}")
            else:
                print("缺失值插值修复成功。")
        
        print("数据完整性检查通过。")
        
        return df

    def _convert_gps_to_ned(self, df):
        """
        将经纬度数据 (Lat, Lon, H) 转换为 NED 坐标 (x, y, z)
        
        Args:
            pd.DataFrame: 清洗后的数据
        
        Returns:
            pd.DataFrame: 坐标转换后的数据
        """
        print("正在进行坐标转换 (LLA -> NED)...")
        
        # 选取第一个点作为参考点 (Home)
        lat0 = np.deg2rad(df['lat'].iloc[0])
        lon0 = np.deg2rad(df['lon'].iloc[0])
        R = 6371000.0   # 地球半径 (m)
        
        # 将当前经纬度转为弧度
        lat_rad = np.deg2rad(df['lat'].values)
        lon_rad = np.deg2rad(df['lon'].values)
        
        # 计算北向距离 x (North)
        df['x'] = R * (lat_rad - lat0)
        
        # 计算东向距离 y (East)
        df['y'] = R * np.cos(lat0) * (lon_rad - lon0)
        
        # 计算地向距离 z (Down)
        df['z'] = -(df['h'].values - self.h0)
        
        return df
    
    def _apply_low_pass_filter(self, df):
        """
        对传感器噪声数据进行零相位低通滤波

        Args:
            pd.DataFrame: 坐标转换后的数据
        
        Returns:
            pd.DataFrame: 低通滤波后的数据
        """
        print("正在对数据进行零相位低通滤波...")
        
        # 滤波器参数
        cutoff_freq = 10.0  # Hz
        nyquist = 0.5 * (1.0 / self.dt)
        normal_cutoff = cutoff_freq / nyquist
        
        # 设计 Butterworth 滤波器
        b, a = scipy.signal.butter(2, normal_cutoff, btype='low', analog=False)
        
        # 需要滤波的列
        filter_cols = ['p', 'q', 'r', 'ax', 'ay', 'az']
        
        # 如果存在气动数据，也建议滤波
        for col in ['TAS', 'Q', 'alpha', 'beta']:
            if col in df.columns:
                filter_cols.append(col)
        
        # 使用 filtfilt 实现零相位滤波
        for col in filter_cols:
            if col in df.columns:
                try:
                    df[col] = scipy.signal.filtfilt(b, a, df[col].values)
                except Exception as e:
                    print(f"警告: 列 {col} 滤波失败: {e}")
                
        return df
    
    def _reconstruct_state_variables(self, df, is_windless):
        """
        状态变量重构，计算机体轴空速分量 uvw。
        
        根据 is_windless 选择计算逻辑：
        - 无风模式：利用地速向量(NED)和欧拉角构建旋转矩阵直接投影。
        - 有风模式：利用气动定义，通过 TAS, alpha, beta 计算。

        Args:
            df (pd.DataFrame): 包含姿态、地速或气动角的输入数据
            is_windless (bool): 是否为无风仿真环境

        Returns:
            pd.DataFrame: 新增了 'u', 'v', 'w' 列的数据
        """
        print("正在进行状态变量重构 (计算 u, v, w)...")
        
        if is_windless:
            # --- 方案 A: 无风仿真环境 (几何投影法) ---
            # 原理: V_body = R_nb^T * V_ned
            
            # 1. 获取欧拉角 (弧度)
            phi = df['phi'].values
            theta = df['theta'].values
            psi = df['psi'].values
            
            # 2. 获取地速 (NED系)
            vn = df['u_g'].values
            ve = df['v_g'].values
            vd = df['w_g'].values
            
            # 3. 预计算三角函数
            c_phi, s_phi = np.cos(phi), np.sin(phi)
            c_th, s_th = np.cos(theta), np.sin(theta)
            c_psi, s_psi = np.cos(psi), np.sin(psi)
            
            # 4. 构建旋转矩阵 R_b_n (NED -> Body) 的元素
            r11 = c_th * c_psi
            r12 = c_th * s_psi
            r13 = -s_th
            
            r21 = s_phi * s_th * c_psi - c_phi * s_psi
            r22 = s_phi * s_th * s_psi + c_phi * c_psi
            r23 = s_phi * c_th
            
            r31 = c_phi * s_th * c_psi + s_phi * s_psi
            r32 = c_phi * s_th * s_psi - s_phi * c_psi
            r33 = c_phi * c_th
            
            # 5. 执行矩阵乘法
            df['u'] = r11*vn + r12*ve + r13*vd
            df['v'] = r21*vn + r22*ve + r23*vd
            df['w'] = r31*vn + r32*ve + r33*vd
            
            # # 补充计算 TAS (如果原数据没有)
            # if 'TAS' not in df.columns:
            #     df['TAS'] = np.sqrt(df['u']**2 + df['v']**2 + df['w']**2)
                
            print("  -> 已通过地速与姿态旋转矩阵计算 u, v, w (无风假设)")
            
        else:
            # --- 方案 B: 有风/实测环境 (气动定义法) ---
            # 原理: 利用定义的 TAS 和气流角分解
            # u = V * cos(a) * cos(b)
            # v = V * sin(b)
            # w = V * sin(a) * cos(b)
            
            tas = df['TAS'].values
            alpha = df['alpha'].values
            beta = df['beta'].values
            
            cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
            cos_beta, sin_beta = np.cos(beta), np.sin(beta)
            
            df['u'] = tas * cos_alpha * cos_beta
            df['v'] = tas * sin_beta
            df['w'] = tas * sin_alpha * cos_beta
            
            print("  -> 已通过 TAS, alpha, beta 计算 u, v, w")

        return df
    
    def _prepare_derivative_labels(self, df, lever_arm=None):
        """
        数值微分和导数监督标签生成。
        
        功能：
        1. 计算角速度的导数 (dot_p, dot_q, dot_r)。
        2. 修正 IMU 加速度数据，将其转换到重心 (CG) 处，作为力 Label。
        
        Args:
            df (pd.DataFrame): 包含 p, q, r, ax, ay, az 的数据
            lever_arm (list/array, optional): [x, y, z] IMU相对于CG的偏移量。
                                              默认为 None，即假设 IMU 安装在重心。

        Returns:
            pd.DataFrame: 新增了 'dot_p', 'dot_q', 'dot_r' 以及修正后的 'ax_cg', 'ay_cg', 'az_cg'
        """
        print("正在计算导数标签与IMU力修正...")
        
        # 1. 计算角加速度 (数值微分)
        for col in ['p', 'q', 'r']:
            dot_col = f'dot_{col}'
            df[dot_col] = np.gradient(df[col].values, self.dt, edge_order=1)
            # df[dot_col] = savgol_filter(
            #     df[col].values, 
            #     window_length = 11,     # 窗口大小 (奇数)
            #     polyorder = 2,          # 多项式阶数
            #     deriv = 1,              # 求一阶导
            #     delta = self.dt         # 自动除以 dt
            # )
            
        # 2. IMU 加速度修正 (杆臂效应)
        ax_imu = df['ax'].values
        ay_imu = df['ay'].values
        az_imu = df['az'].values
        
        if lever_arm is not None and not np.allclose(lever_arm, 0):
            r = np.array(lever_arm) # (3,)
            print(f"  -> 检测到力臂设置 {r}，正在应用杆臂效应修正...")
            
            omega = df[['p', 'q', 'r']].values # (N, 3)
            dot_omega = df[['dot_p', 'dot_q', 'dot_r']].values # (N, 3)
            
            # a_CG = a_IMU - dot_omega x r - omega x (omega x r) 
            term1 = np.cross(dot_omega, r) # (N, 3)
            omega_x_r = np.cross(omega, r)
            term2 = np.cross(omega, omega_x_r) # (N, 3)
            
            a_cg = df[['ax', 'ay', 'az']].values - term1 - term2
            df['ax_cg'] = a_cg[:, 0]
            df['ay_cg'] = a_cg[:, 1]
            df['az_cg'] = a_cg[:, 2]
            
        else:
            print("  -> 未检测到力臂数据，忽略杆臂效应。")
            df['ax_cg'] = ax_imu
            df['ay_cg'] = ay_imu
            df['az_cg'] = az_imu
            
        return df
    
    def _normalize_data(self, df, save_path):
        """
        数据归一化
        
        功能：
        1. 实例化 FlightDataScaler。
        2. fit: 计算并保存统计量 (mean, std, min, max)。
        3. transform: 将数据转换为归一化数值。
        4. 保存 scaler 对象到磁盘，供后续训练使用。
        
        Args:
            df (pd.DataFrame): 包含物理单位的完整数据
            save_path (str): 保存 scaler 的路径 (如 'scaler.pkl')

        Returns:
            pd.DataFrame: 全量归一化后的数据
        """
        print(f"正在进行数据归一化 (目标路径: {save_path})...")
        
        # 1. 实例化独立的 Scaler 类
        scaler = FlightDataScaler()
        
        # 2. 计算统计参数 (Fit)
        scaler.fit(df)
        
        # 3. 保存参数 (Save)
        scaler.save(save_path)
        
        # 4. 执行转换 (Transform)
        # 注意：这里返回一个新的 DataFrame，保留原 df 可能用于对比分析
        df_norm = scaler.transform(df)
        
        return df_norm
    
    def _create_trajectory_slices(self, df_raw, df_norm, window_size, stride):
        """
        轨迹切片与数据集构造。
        
        功能：
        使用滑动窗口将连续的飞行数据切割成多个片段。
        
        Args:
            df_raw (pd.DataFrame): 原始物理数据 (用于切 Labels 和 GT)
            df_norm (pd.DataFrame): 归一化数据 (用于切 NN Inputs 和 x0)
            window_size (int): 窗口长度 (Time_Steps)
            stride (int): 滑动步长 (决定切片的重叠程度)

        Returns:
            dict: 训练数据字典
                - 'input_ode_x0': (N, 12) 归一化的初始状态
                - 'input_nn_controls': (N, T, 4) 归一化的控制量序列
                - 'labels': 包含物理真值的字典
                    - 'traj': (N, T, 12) 物理状态轨迹真值
                    - 'force': (N, T, 6) 物理力/导数真值
        """
        print(f"正在进行轨迹切片 (窗口: {window_size}, 步长: {stride})...")
        
        # 1. 定义列名列表
        # 状态量 (12维)
        state_cols = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z']
        
        # 控制量 (4维)
        power_col = 'rpm' if 'rpm' in df_raw.columns else 'delta_t'
        control_cols = ['delta_e', 'delta_a', 'delta_r', power_col]
        
        # 导数监督/Label (6维)
        label_cols = ['ax_cg', 'ay_cg', 'az_cg', 'dot_p', 'dot_q', 'dot_r']
        
        # 2. 提取数据为 Numpy 数组
        # (1) 归一化源 (用于模型输入)
        data_states_norm = df_norm[state_cols].values
        data_controls_norm = df_norm[control_cols].values

        # (2) 物理源 (用于真值监督)
        data_states_real = df_raw[state_cols].values
        data_labels_real = df_raw[label_cols].values
        time_arr = df_raw['time'].values
        t_tolerance = 1e-4
        
        total_len = len(df_raw)
        
        # 3. 滑动窗口切片
        slices = {
            'input_ode_x0': [],       # (N, 12)
            'input_nn_controls': [],  # (N, T, 4)
            'label_traj': [],         # (N, T, 12)
            'label_force': [],        # (N, T, 6)
            'states_norm': []         # (N, T, 12)
        }
        
        for i in range(0, total_len - window_size + 1, stride):
            idx_end = i + window_size
            
            # 连续性检查
            t_slice = time_arr[i : idx_end]
            t_span = t_slice[-1] - t_slice[0]
            expected_span = (window_size - 1) * self.dt
            
            if abs(t_span - expected_span) > t_tolerance * window_size:
                continue
            
            # (1) 输入给 ODE 的初值 x0 (只需取窗口第1帧)
            slices['input_ode_x0'].append(data_states_norm[i])
            
            # (2) 输入给 NN 的控制量上下文 (取整个窗口序列)
            slices['input_nn_controls'].append(data_controls_norm[i : idx_end])
            
            # (3) 物理真值 (用于 Loss)
            slices['label_traj'].append(data_states_real[i : idx_end])
            slices['label_force'].append(data_labels_real[i : idx_end])

            # (4) 归一化状态 (用于监督阶段模型输入)
            slices['states_norm'].append(data_states_norm[i : idx_end])
            
        # 4. 堆叠为三维数组 (N, T, D)
        if len(slices['input_ode_x0']) == 0:
            raise ValueError("数据长度不足，无法生成切片")
            
        dataset = {
            'input_ode': np.array(slices['input_ode_x0'], dtype=np.float32),
            
            'input_nn': np.array(slices['input_nn_controls'], dtype=np.float32), 
            
            'labels': {
                'traj': np.array(slices['label_traj'], dtype=np.float32),
                'force': np.array(slices['label_force'], dtype=np.float32)
            },

            'states_norm': np.array(slices['states_norm'], dtype=np.float32)
        }
        
        return dataset
    
    def visualize_comparison(self, df_raw, df_proc, save_dir='./viz_results'):
        """
        可视化对比处理前后的数据
        Args:
            df_raw (DataFrame): 原始数据 (滤波/Unwrap前)
            df_proc (DataFrame): 处理后的数据
            save_dir (str): 图片保存目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"正在生成可视化对比图，保存至 {save_dir} ...")

        # 定义分组
        groups = {
            # 组 1: 角度与角速率 (重点检查 Unwrap 和 滤波)
            'Attitude_Rates': {
                'cols': ['phi', 'theta', 'psi', 'p', 'q', 'r'],
                'layout': (2, 3),
                'figsize': (18, 10)
            },
            # 组 2: 速度与加速度 (重点检查 IMU 降噪)
            # 注意: u,v,w 是重构量，如果 raw 里没有，代码会自动跳过
            'Velocity_Accel': {
                'cols': ['TAS', 'alpha', 'beta', 'ax_cg', 'ay_cg', 'az_cg'],
                'layout': (2, 3),
                'figsize': (18, 10)
            },
            # 组 3: 控制量
            'Controls': {
                'cols': ['delta_e', 'delta_a', 'delta_r', 'rpm', 'delta_t'], # 兼容不同列名
                'layout': (2, 3),
                'figsize': (18, 10)
            }
        }

        # 绘图循环
        for group_name, config in groups.items():
            cols = [c for c in config['cols'] if c in df_proc.columns]
            if not cols:
                continue
                
            rows, grid_cols = config['layout']
            fig, axes = plt.subplots(rows, grid_cols, figsize=config['figsize'])
            axes = axes.flatten()
            
            # 时间轴
            time = np.arange(len(df_proc)) * self.dt 
            
            for i, col in enumerate(cols):
                ax = axes[i]
                
                # 1. 绘制原始数据 (灰色、透明、虚线)
                # 只有当原始数据也有这一列时才画 (应对 u,v,w 这种重构变量)
                if col in df_raw.columns:
                    ax.plot(time, df_raw[col].values, color='gray', alpha=0.5, 
                            linestyle='--', label='Raw (Noisy/Wrapped)')
                
                # 2. 绘制处理后数据 (蓝色/红色、实线)
                # 对 psi 特殊处理颜色，醒目一点
                color = 'red' if col == 'psi' else 'tab:blue'
                ax.plot(time, df_proc[col].values, color=color, linewidth=1.5, 
                        label='Processed (Clean)')
                
                ax.set_title(col)
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='upper right', fontsize='small')
                
                # 仅在最后一行显示 x 轴标签
                if i >= (rows - 1) * grid_cols:
                    ax.set_xlabel('Time (s)')

            # 隐藏多余的子图
            for j in range(len(cols), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'{group_name}_comparison.png')
            plt.savefig(save_path, dpi=150)
            # plt.show()
            plt.close(fig)
            
        print("可视化完成。")


# 可视化测试代码
if __name__ == "__main__":
    TEST_CONFIG = {
        'paths': {
            'raw_csv': 'Document41.csv',
            'scaler': 'scaler41.pkl',
            'dataset': 'dataset41.pt'
        },
        'preprocess': {
            'is_windless': True,
            'lever_arm': None,
            'visualize': True
        },
        'data': {
            'window_size': 100,
            'stride': 20,
            'dt': 0.02,
        },
        'props': {
            'h0': 1100.5,
        }
    }

    pipeline = FlightDataPreprocessor()

    try:
        data_dict = pipeline.run_pipeline(TEST_CONFIG)
        
        print("调试运行成功！")
        print("请查看当前目录下的 'viz_results' 文件夹以检查可视化图表。")
        
    except FileNotFoundError as e:
        print(f"\n[错误] 找不到文件: {e}")
        print(f"请检查 {TEST_CONFIG['paths']['raw_csv']} 是否在当前目录下。")
    except Exception as e:
        print(f"\n[错误] 运行中发生异常: {e}")