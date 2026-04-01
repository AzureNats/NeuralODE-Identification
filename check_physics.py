import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

class FlightDataValidator:
    def __init__(self, dt=0.02, angles_in_degrees=False):
        """
        飞行数据物理一致性校验器
        Args:
            dt: 采样时间间隔 (s)
            angles_in_degrees: CSV里的角度是否为角度制(deg)。如果是rad则设为False。
        """
        self.dt = dt
        self.mass = 420.0
        self.angles_in_degrees = angles_in_degrees

    def run_validation(self, csv_path, save_dir='./validation_plots'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"正在读取数据进行物理一致性校验: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 1. 姿态运动学校验 (欧拉角微分 vs 角速度)
        self._check_attitude_kinematics(df, save_dir)
        
        # 2. 速度运动学校验 (风三角与地速投影)
        self._check_velocity_kinematics(df, save_dir)
        
        # 3. 动力学校验 (比力 vs 气动力+推力)
        self._check_dynamics(df, save_dir)
        
        print(f"校验完成！请前往 {save_dir} 查看对比图。")

    def _get_angle_rad(self, series):
        """处理角度单位，确保输出为弧度"""
        if self.angles_in_degrees:
            return np.deg2rad(series.values)
        return series.values

    def _check_attitude_kinematics(self, df, save_dir):
        """
        验证: 欧拉角的导数 是否等于 由机体角速度转换来的角变率
        """
        print(" -> 正在进行 [姿态运动学] 校验...")
        time = df['time'].values
        
        phi = self._get_angle_rad(df['phi'])
        theta = self._get_angle_rad(df['theta'])
        psi = self._get_angle_rad(df['psi'])
        
        # 假设 p, q, r 已经是 rad/s
        p = df['p'].values
        q = df['q'].values
        r = df['r'].values
        
        # 左边：对记录的欧拉角直接求导 (使用 SG 滤波器平滑求导)
        dot_phi_num = savgol_filter(phi, window_length=11, polyorder=2, deriv=1, delta=self.dt)
        dot_theta_num = savgol_filter(theta, window_length=11, polyorder=2, deriv=1, delta=self.dt)
        dot_psi_num = savgol_filter(psi, window_length=11, polyorder=2, deriv=1, delta=self.dt)
        
        # 右边：根据运动学方程用 p,q,r 计算预期的欧拉角变化率
        tan_theta = np.tan(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        
        dot_phi_eq = p + (q * sin_phi + r * cos_phi) * tan_theta
        dot_theta_eq = q * cos_phi - r * sin_phi
        dot_psi_eq = (q * sin_phi + r * cos_phi) / (cos_theta + 1e-6) # 防止除零
        
        # 绘图对比
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Attitude Kinematics Consistency\n(Numerical Derivative vs Kinematic Equation)', fontsize=14)
        
        labels = [('d(Phi)/dt', dot_phi_num, dot_phi_eq), 
                  ('d(Theta)/dt', dot_theta_num, dot_theta_eq), 
                  ('d(Psi)/dt', dot_psi_num, dot_psi_eq)]
                  
        for i, (name, num, eq) in enumerate(labels):
            axes[i].plot(time, num, label='Derivative of Euler Angle (Left)', color='gray', linestyle='--')
            axes[i].plot(time, eq, label='Computed from p,q,r (Right)', color='blue', alpha=0.7)
            axes[i].set_ylabel(f'{name} (rad/s)')
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
            
        axes[2].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '1_Attitude_Kinematics.png'), dpi=150)
        plt.close()

    def _check_velocity_kinematics(self, df, save_dir):
        """
        验证: 基于TAS和气流角算出的空速(u,v,w) 是否等于 经纬高算出的地速投影(无风假设)
        """
        print(" -> 正在进行 [速度与风三角] 校验...")
        time = df['time'].values
        
        # 1. 从气动数据计算机体系空速
        tas = df['TAS'].values
        alpha = self._get_angle_rad(df['alpha'])
        beta = self._get_angle_rad(df['beta'])
        
        u_air = tas * np.cos(alpha) * np.cos(beta)
        v_air = tas * np.sin(beta)
        w_air = tas * np.sin(alpha) * np.cos(beta)
        
        # 2. 从地速计算投影到机体系的几何速度 (NED -> Body)
        u_g = df['u_g'].values
        v_g = df['v_g'].values
        w_g = df['w_g'].values
        
        phi = self._get_angle_rad(df['phi'])
        theta = self._get_angle_rad(df['theta'])
        psi = self._get_angle_rad(df['psi'])
        
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        u_gnd = (c_th * c_psi) * u_g + (c_th * s_psi) * v_g + (-s_th) * w_g
        v_gnd = (s_phi * s_th * c_psi - c_phi * s_psi) * u_g + (s_phi * s_th * s_psi + c_phi * c_psi) * v_g + (s_phi * c_th) * w_g
        w_gnd = (c_phi * s_th * c_psi + s_phi * s_psi) * u_g + (c_phi * s_th * s_psi - s_phi * c_psi) * v_g + (c_phi * c_th) * w_g

        # 绘图对比
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Velocity Kinematics Consistency (Windless Assumption)\n(Aero Airspeed vs Ground Speed Projection)', fontsize=14)
        
        labels = [('Body U (m/s)', u_air, u_gnd), 
                  ('Body V (m/s)', v_air, v_gnd), 
                  ('Body W (m/s)', w_air, w_gnd)]
                  
        for i, (name, air, gnd) in enumerate(labels):
            axes[i].plot(time, gnd, label='Projected from Ground Speed (NED)', color='gray', linestyle='--')
            axes[i].plot(time, air, label='Calculated from TAS, Alpha, Beta', color='green', alpha=0.7)
            axes[i].set_ylabel(name)
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
            
        axes[2].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '2_Velocity_Kinematics.png'), dpi=150)
        plt.close()

    def _check_dynamics(self, df, save_dir):
        """
        验证: 质量 * 比力(IMU) 是否等于 真值气动力 + 真值推力
        """
        print(" -> 正在进行 [动力学 F=ma] 校验...")
        time = df['time'].values
        
        # 质量 * IMU加速度计 = 实际受到的非引力外力
        force_x_imu = df['ax'].values * self.mass
        force_y_imu = df['ay'].values * self.mass
        force_z_imu = df['az'].values * self.mass
        
        # 仿真器记录的气动真值
        fx_aero = df['FX'].values
        fy_aero = df['FY'].values
        fz_aero = df['FZ'].values

        # 获取气流角 (保证是弧度)
        alpha = self._get_angle_rad(df['alpha'])
        beta = self._get_angle_rad(df['beta'])

        # 预计算三角函数
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)

        # 根据 R_bw = R_y(alpha) * R_z(-beta) 进行矩阵相乘投影
        fx_aero_body = fx_aero * cos_a * cos_b - fy_aero * cos_a * sin_b - fz_aero * sin_a
        fy_aero_body = fx_aero * sin_b + fy_aero * cos_b
        fz_aero_body = fx_aero * sin_a * cos_b - fy_aero * sin_a * sin_b + fz_aero * cos_a
        
        deltp = np.clip(df['delta_t'].values, 0, 100)
        Tdeltp_node = np.array([0, 80, 100])
        Thrust_data = np.array([0, 982, 1586])
        simulink_thrust = np.interp(deltp, Tdeltp_node, Thrust_data)

        # 绘图对比
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Dynamics Consistency (Force vs IMU Specific Force)\nAssuming Mass = {self.mass} kg', fontsize=14)
        
        labels = [('X-axis Force (N)', force_x_imu, fx_aero_body, 'Fx_body + Thrust'), 
                  ('Y-axis Force (N)', force_y_imu, fy_aero_body, 'Fy_body'), 
                  ('Z-axis Force (N)', force_z_imu, fz_aero_body, 'Fz_body')]
                  
        for i, (name, imu_f, aero_f, note) in enumerate(labels):
            axes[i].plot(time, imu_f, label='Mass * IMU Accel (Total External Force)', color='gray', linestyle='--')
            axes[i].plot(time, aero_f, label='Aero Force in Body Axes (Transformed)', color='red', alpha=0.7)
            
            # 计算两者的差值（发动机推力）
            diff = imu_f - aero_f
            axes[i].plot(time, diff, label='Difference (Likely Thrust)', color='purple', alpha=0.5)
            if i == 0:
                axes[i].plot(time, simulink_thrust, label='Simulink Engine Model (P)', color='orange', linestyle=':', linewidth=2.5)
            axes[i].set_ylabel(name)
            axes[i].legend(loc='upper right')
            axes[i].grid(True)
            
        axes[2].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '3_Dynamics.png'), dpi=150)
        plt.close()

if __name__ == "__main__":
    CSV_FILE = "Document41.csv"
    # 如果原始CSV中的欧拉角和气流角是角度度数(deg)，设为 True；如果是弧度(rad)，设为 False
    ANGLES_IN_DEGREES = False 
    
    validator = FlightDataValidator(dt=0.02, angles_in_degrees=ANGLES_IN_DEGREES)
    validator.run_validation(CSV_FILE)