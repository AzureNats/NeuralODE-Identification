import torch
import torch.nn as nn

class CoefficientNet(nn.Module):
    def __init__(self):
        """
        神经网络模块,用于预测气动系数

        架构:
            Input(10) -> Linear(64) -> Tanh 
                      -> Linear(128) -> Tanh 
                      -> Linear(64) -> Tanh 
                      -> Linear(6) -> Output
        """
        super().__init__()
        
        # 定义网络层
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.Tanh(),
            
            nn.Linear(64, 128),
            nn.Tanh(),
            
            nn.Linear(128, 64),
            nn.Tanh(),
            
            nn.Linear(64, 6)
        )
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化，适用于 Tanh 激活函数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        with torch.no_grad():
            self.net[-1].weight.data *= 0.01 
            self.net[-1].bias.data.fill_(0.0)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 归一化的输入向量 (Batch_Size, 10)
                        包含 [u, v, w, p, q, r, de, da, dr, dt]
        
        Returns:
            out (Tensor): 预测的气动系数 (Batch_Size, 6)
                          包含 [Cx, Cy, Cz, Cl, Cm, Cn]
        """
        return self.net(x)


class AerialSystemODE(nn.Module):
    def __init__(self, neural_net, known_props, scaler, device):
        """
        物理系统 ODE 模块
        
        Args:
            neural_net (nn.Module): 负责预测气动系数的神经网络 (CoefficientNet)
            known_props (dict): 物理属性字典 {m, S, b, c, I_inv, ...}
            scaler (FlightDataScaler): 用于实时归一化/反归一化的工具
        """
        super().__init__()
        
        self.net = neural_net
        self.props = known_props
        self.scaler = scaler
        
        # 定义状态量和控制量的顺序 (必须与 Dataset 切片一致)
        self.state_keys = ['u', 'v', 'w', 'p', 'q', 'r', 
                           'phi', 'theta', 'psi', 'x', 'y', 'z']
        
        # 自动推断控制量 key (兼容 rpm/delta_t)
        # 检查 scaler 里到底存的是 rpm 还是 delta_t
        power_key = 'rpm' if 'rpm' in scaler.stats else 'delta_t'
        self.control_keys = ['delta_e', 'delta_a', 'delta_r', power_key]

        # 上下文缓存
        # 这些变量将在积分开始前由 set_control_context 填充
        # 用于在 forward 过程中根据时间 t 查询对应的控制量
        self._ctx_times = None      # 时间网格 (T,)
        self._ctx_controls = None   # 控制量序列 (Batch, T, 4) [归一化数值]

        # 物理参数预处理
        self.mass = torch.tensor(known_props['m']).float().to(device)
        self.S = torch.tensor(known_props['S']).float().to(device)
        self.b = torch.tensor(known_props['b']).float().to(device)
        self.c = torch.tensor(known_props['c']).float().to(device)
        self.base_alt = torch.tensor(known_props.get('base_altitude', 0.0)).float().to(device)
        self.I = torch.tensor(known_props['I']).float().to(device)
        self.I_inv = torch.inverse(self.I)
        self.T_offset = torch.tensor(known_props['T_offset']).float().to(device)

        # ISA 大气模型常数
        self.register_buffer('g', torch.tensor(9.80665).to(device))   # 重力加速度 (m/s^2)
        self.register_buffer('rho_0', torch.tensor(1.225).to(device)) # 空气密度 (kg/m^3)
        self.register_buffer('T0', torch.tensor(288.15).to(device))   # 海平面温度 (K)
        self.register_buffer('L', torch.tensor(0.0065).to(device))    # 温度递减率 (K/m)
        self.register_buffer('R', torch.tensor(287.05).to(device))    # 气体常数 (J/kg/K)

        # # 动力系统拟合系数 c_H_V
        # # T = p[0]*rpm^2 + p[1]*rpm + p[2]
        # self.register_buffer('c_0_15', torch.tensor([4.71865251e-05, -7.27991513e-02, 1.70176257e+02]).to(device))
        # self.register_buffer('c_0_35', torch.tensor([3.96509480e-05, -1.24969118e-01, 2.21010645e+02]).to(device))
        # self.register_buffer('c_3_15', torch.tensor([2.81943710e-05, 3.29894820e-02, -1.68150701e+01]).to(device))
        # self.register_buffer('c_3_35', torch.tensor([2.59178488e-05, -5.41092634e-02, 9.97308307e+01]).to(device))

    def set_control_context(self, times, controls):
        """
        在调用 odeint 之前必须调用此函数，注入当前的控制量序列。
        
        Args:
            times (Tensor): 时间序列 (T,)，例如 [0.00, 0.02, 0.04...]
            controls (Tensor): 归一化的控制量 (Batch, T, 4)
        """
        self._ctx_times = times
        self._ctx_controls = controls

    def get_air_density(self, z_ned):
        """
        根据 ISA 模型计算当前高度的大气密度
        
        Args:
            z_ned (Tensor): NED 坐标系下的 z 轴位置 (向下为正, m)
                            Alt = Base_Alt - z_ned
        Returns:
            rho (Tensor): 当前密度 (kg/m^3)
        """
        # 1. 计算绝对海拔高度 (m)
        alt = self.base_alt - z_ned
        
        # 2. 计算温度
        alt = torch.clamp(alt, min=-1000.0, max=11000.0)
        T = self.T0 - self.L * alt
        
        # 3. 计算密度
        exponent = (self.g / (self.R * self.L)) - 1.0
        rho = self.rho_0 * torch.pow((T / self.T0), exponent)
        
        return rho

    def get_control_at_t(self, t):
        """
        线性插值获取任意时刻 t 的控制量 u(t)
        
        Args:
            t (Tensor): 当前积分时间 (标量 float 或 0-d Tensor)
            
        Returns:
            u_t (Tensor): 插值后的控制量 (Batch, 4) [归一化数值]
        """
        # 1. 边界处理
        # 防止浮点误差导致 t 略微超出范围
        t = torch.clamp(t, min=self._ctx_times[0], max=self._ctx_times[-1])
        
        # 2. 查找索引
        # 找到 t 在时间网格中的位置: times[idx] <= t < times[idx+1]
        # side='right' 返回满足 times[i] <= t 的最后一个索引 + 1
        idx = torch.searchsorted(self._ctx_times, t, side='right') - 1
        
        # 防止 idx 越界 (针对 t 正好等于最后一个时间点的情况)
        idx = torch.clamp(idx, 0, len(self._ctx_times) - 2)
        
        # 3. 取出相邻的两个时间点
        t0 = self._ctx_times[idx]
        t1 = self._ctx_times[idx + 1]
        
        # 4. 计算插值权重 alpha (0~1)
        alpha = (t - t0) / (t1 - t0 + 1e-9)
        
        # 5. 取出相邻的控制量 (Batch, 4)
        # _ctx_controls 形状是 (Batch, T, 4)，需要在 dim=1 上取索引
        u0 = self._ctx_controls[:, idx, :]
        u1 = self._ctx_controls[:, idx + 1, :]
        
        # 6. 线性插值
        u_t = u0 + alpha * (u1 - u0)
        
        return u_t
    
    # def calculate_thrust(self, z_ned, airspeed_mps, rpm):
    #     """
    #     基于工况点插值计算推力
        
    #     Args:
    #         z_ned (Tensor): NED 坐标系下的 z 轴位置 (向下为正, km)
    #         airspeed_mps (Tensor): 空速 (m/s)
    #         rpm (Tensor): 发动机转速 (rpm)
            
    #     Returns:
    #         thrust (Tensor): 推力 (N)
    #     """
    #     # 1. 计算绝对海拔高度 (km)
    #     alt = (self.base_alt - z_ned) / 1000.0
        
    #     # 2. 辅助函数: 多项式计算 ax^2 + bx + c
    #     def poly_val(coeffs, x):
    #         return coeffs[0] * (x ** 2) + coeffs[1] * x + coeffs[2]

    #     # 3. 计算四个基准工况点在当前 RPM 下的推力
    #     t_0_15 = poly_val(self.c_0_15, rpm)
    #     t_0_35 = poly_val(self.c_0_35, rpm)
    #     t_3_15 = poly_val(self.c_3_15, rpm)
    #     t_3_35 = poly_val(self.c_3_35, rpm)

    #     # 4. 计算插值权重
    #     # 速度权重 w_v (15~35 m/s)
    #     w_v = (airspeed_mps - 15.0) / (35.0 - 15.0)
    #     # w_v = torch.clamp(w_v, -0.5, 1.5) # 可选: 限制外推程度
        
    #     # 高度权重 w_h (0~3 km)
    #     w_h = (alt - 0.0) / (3.0 - 0.0)
    #     # w_h = torch.clamp(w_h, -0.2, 1.2) # 可选: 限制外推程度

    #     # 5. 双线性插值
    #     t_h0 = torch.lerp(t_0_15, t_0_35, w_v)
    #     t_h3 = torch.lerp(t_3_15, t_3_35, w_v)
    #     thrust = torch.lerp(t_h0, t_h3, w_h)
        
    #     return torch.max(thrust, torch.tensor(0.0).to(thrust.device))

    def calculate_thrust(self, delta_t):
        """
        基于 Simulink 三点插值的推力模型
        Args:
            delta_t (Tensor): 油门指令 (0~100)
        Returns:
            thrust (Tensor): X轴推力 (N)
        """
        dt_val = torch.clamp(delta_t, min=0.0, max=100.0)
        
        # 向量化三点插值
        # 节点: (0, 0), (80, 982), (100, 1586)
        thrust = torch.where(
            dt_val <= 80.0,
            12.275 * dt_val,                          # 0~80 段斜率: 982/80 = 12.275
            982.0 + 30.2 * (dt_val - 80.0)            # 80~100 段斜率: (1586-982)/20 = 30.2
        )
        
        return thrust

    def physics_equations(self, state_real, controls_real, aero_coeffs):
        """
        受力求解与 6-DOF 刚体动力学方程
        
        Args:
            state_real (Tensor): (B, 12) [u, v, w, p, q, r, phi, theta, psi, x, y, z]
            controls_real (Tensor): (B, 4) [de, da, dr, dt/rpm]
            aero_coeffs (Tensor): (B, 6) [Cx, Cy, Cz, Cl, Cm, Cn]
            
        Returns:
            dx_dt (Tensor): (B, 12) 状态导数
            diagnostic (dict): 包含中间物理量 (Forces, Moments) 用于 Loss 诊断
        """
        # 1. 数据解包
        # 速度 (机体轴)
        vel_vec = state_real[:, 0:3]
        u, v, w = vel_vec[:, 0], vel_vec[:, 1], vel_vec[:, 2]
        
        # 角速度 (机体轴)
        omega_vec = state_real[:, 3:6]
        p, q, r = omega_vec[:, 0], omega_vec[:, 1], omega_vec[:, 2]
        
        # 姿态角 (欧拉角)
        euler_vec = state_real[:, 6:9]
        theta_clamped = torch.clamp(euler_vec[:, 1], min=-1.48, max=1.48)
        phi, theta, psi = euler_vec[:, 0], theta_clamped, euler_vec[:, 2]
        
        # 气动系数
        Cx, Cy, Cz = aero_coeffs[:, 0], aero_coeffs[:, 1], aero_coeffs[:, 2]
        Cl, Cm, Cn = aero_coeffs[:, 3], aero_coeffs[:, 4], aero_coeffs[:, 5]

        # 高度与转速
        z = state_real[:, 11]
        delta_t = controls_real[:, 3]
        
        # 2. 计算动压
        rho = self.get_air_density(z)
        # Q = 0.5 * rho * V^2
        V_sq = torch.sum(vel_vec ** 2, dim = 1)
        Q = 0.5 * rho * V_sq
        
        # 3. 计算气动力与力矩
        q_s = Q * self.S

        # F_aero = Q * S * [Cx, Cy, Cz]
        Fx_aero = q_s * Cx
        Fy_aero = q_s * Cy
        Fz_aero = q_s * Cz
        F_aero_vec = torch.stack([Fx_aero, Fy_aero, Fz_aero], dim=1)
        
        # M_aero = Q * S * [b*Cl, c*Cm, b*Cn]
        L_aero = q_s * self.b * Cl
        M_aero = q_s * self.c * Cm
        N_aero = q_s * self.b * Cn
        M_aero_vec = torch.stack([L_aero, M_aero, N_aero], dim=1)
        
        # 4. 计算推力
        # V_total = torch.sqrt(V_sq + 1e-9)
        # Fx_thrust = self.calculate_thrust(z, V_total, rpm)
        Fx_thrust = self.calculate_thrust(delta_t)

        # 构造推力向量 F_thrust_vec = [T, 0, 0] # (B, 3)
        zeros = torch.zeros_like(Fx_thrust)
        F_thrust_vec = torch.stack([Fx_thrust, zeros, zeros], dim=1)
        
        # 计算推力产生的力矩 (Moment = r x F) # (B, 3)
        # r = t_offset (需扩展维度以匹配 Batch) (3,) -> (B, 3)
        r_vec = self.T_offset.unsqueeze(0).expand(state_real.shape[0], -1)
        M_thrust_vec = torch.cross(r_vec, F_thrust_vec, dim=1)
        
        # 5. 计算重力
        # 将重力矢量从 NED 转换到 Body
        # F_g_body = R_nb^T * [0, 0, mg]^T
        mg = self.mass * self.g
        s_phi, c_phi = torch.sin(phi), torch.cos(phi)
        s_theta, c_theta = torch.sin(theta), torch.cos(theta)
        
        Fx_grav = -mg * s_theta
        Fy_grav =  mg * c_theta * s_phi
        Fz_grav =  mg * c_theta * c_phi
        F_grav_vec = torch.stack([Fx_grav, Fy_grav, Fz_grav], dim=1)
        
        # 6. 合力与合力矩 # (B, 3)
        F_non_grav_vec = F_aero_vec + F_thrust_vec
        F_vec = F_non_grav_vec + F_grav_vec
        M_vec = M_aero_vec + M_thrust_vec
        
        # 7. 动力学方程求解 (Dynamics)
        # 7.1 线运动: dot_v = F/m - omega x v
        cross_prod_v = torch.cross(omega_vec, vel_vec, dim=1)
        dot_vel = (F_vec / self.mass) - cross_prod_v
        
        # 7.2 角运动: dot_omega = I_inv * (M - omega x (I * omega))
        # 陀螺力矩项: omega x (I * omega)
        I_omega = omega_vec @ self.I.t()
        gyro_moment = torch.cross(omega_vec, I_omega, dim=1)
        
        # 欧拉方程
        # dot_omega = I_inv @ (M_vec - gyro_moment)
        net_moment = M_vec - gyro_moment
        dot_omega = net_moment @ self.I_inv.t()
        
        # 8. 运动学方程求解 (Kinematics)
        # 8.1 姿态角导数: dot_Euler = W * omega
        tan_theta = s_theta / (c_theta + 1e-9)
        dot_phi   = p + (q * s_phi + r * c_phi) * tan_theta
        dot_theta = q * c_phi - r * s_phi
        dot_psi   = (q * s_phi + r * c_phi) / (c_theta + 1e-9)
        
        dot_att = torch.stack([dot_phi, dot_theta, dot_psi], dim=1)
        
        # 8.2 位置导数 (地速): dot_Pos = R_nb * v
        # 将速度从 Body 转换回 NED
        c_psi, s_psi = torch.cos(psi), torch.sin(psi)
        
        # R_nb 矩阵乘法展开
        # dot_x (North)
        dot_x = c_theta*c_psi*u + \
                (s_phi*s_theta*c_psi - c_phi*s_psi)*v + \
                (c_phi*s_theta*c_psi + s_phi*s_psi)*w
        
        # dot_y (East)
        dot_y = c_theta*s_psi*u + \
                (s_phi*s_theta*s_psi + c_phi*c_psi)*v + \
                (c_phi*s_theta*s_psi - s_phi*c_psi)*w
                
        # dot_z (Down)
        dot_z = -s_theta*u + \
                s_phi*c_theta*v + \
                c_phi*c_theta*w
                
        dot_pos = torch.stack([dot_x, dot_y, dot_z], dim=1)
        
        # 9. 打包输出 (B, 12)
        dx_dt = torch.cat([dot_vel, dot_omega, dot_att, dot_pos], dim=1)
        
        # 诊断信息
        accel_proper = F_non_grav_vec / self.mass 
        pred_force = torch.cat([accel_proper, dot_omega], dim=1)  # (B, 6)

        diagnostic = {
            'F_total': F_vec,
            'M_total': M_vec,
            'accel_body': dot_vel,        # 真实的运动学加速度 (dv/dt)
            'pred_force': pred_force      # 比力与角加速度，用于计算 Loss_force
        }
        
        return dx_dt, diagnostic

    def forward(self, t, state_norm):
        """
        标准 ODE 接口: dy/dt = f(t, y)
        供 torchdiffeq.odeint 调用。
        
        Args:
            t (Tensor): 当前时间 (标量)
            state_norm (Tensor): 归一化的状态 (B, 12) 
                                 [u, v, w, p, q, r, phi, theta, psi, x, y, z]
        
        Returns:
            d_state_norm (Tensor): 归一化的状态导数 (B, 12)
        """
        # 1. 获取控制量 (B, 4)
        u_norm = self.get_control_at_t(t)
        
        # 2. 神经网络输入 (B, 10)
        # NN 输入 = [归一化状态的前6维(u,v,w,p,q,r), 归一化控制(4维)]
        nn_input = torch.cat([state_norm[:, :6], u_norm], dim=1)
        
        # 3. 神经网络预测 (B, 6)
        aero_coeffs = self.net(nn_input)
        
        # 4. 反归一化
        state_real = self.scaler.inverse_transform_vector(state_norm, self.state_keys)
        u_real = self.scaler.inverse_transform_vector(u_norm, self.control_keys)

        # 5. 代入物理方程 (B, 12)
        dx_dt_real, _ = self.physics_equations(state_real, u_real, aero_coeffs)
        
        # 6. 导数归一化
        # d(norm)/dt = d(real)/dt * scale_factor
        dx_dt_norm = self.scaler.scale_derivative_vector(dx_dt_real, self.state_keys)

        return torch.clamp(dx_dt_norm, min=-10.0, max=10.0)
    
    def predict_forces_and_moments(self, state_norm, u_norm):
        """
        直接输入给定的状态和控制，计算物理力和力矩。
        用于计算: Loss_force = || F_pred/m - a_IMU ||
        
        Args:
            state_norm (Tensor): (B, 12) 或 (B, T, 12)
            u_norm (Tensor): (B, 4) 或 (B, T, 4)
            
        Returns:
            diagnostic (dict): 包含 'F_total', 'M_total', 'accel_body' 'pred_force'等物理量
        """
        # 支持序列输入 (B, T, D) -> 展平为 (B*T, D) 处理，最后再变回来
        original_shape = None
        if state_norm.dim() == 3:
            B, T, D_s = state_norm.shape
            _, _, D_u = u_norm.shape
            original_shape = (B, T)
            state_norm = state_norm.reshape(B*T, D_s)
            u_norm = u_norm.reshape(B*T, D_u)
            
        # 1. 神经网络预测
        nn_input = torch.cat([state_norm[:, :6], u_norm], dim=1)
        aero_coeffs = self.net(nn_input)
        
        # 2. 反归一化
        state_real = self.scaler.inverse_transform_vector(state_norm, self.state_keys)
        u_real = self.scaler.inverse_transform_vector(u_norm, self.control_keys)
            
        # 3. 物理计算
        _, diagnostic = self.physics_equations(state_real, u_real, aero_coeffs)
        
        # 4. 形状恢复：如果输入是序列，将结果 reshape 回 (B, T, D)
        if original_shape is not None:
            B, T = original_shape
            for k, v in diagnostic.items():
                # v 原本是 (B*T, 3) -> (B, T, 3)
                diagnostic[k] = v.reshape(B, T, -1)
                
        return diagnostic




# --- 单元测试代码 ---
if __name__ == "__main__":
    pass