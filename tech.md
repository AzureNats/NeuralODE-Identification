# 基于Neural ODE的飞行器动力学模型辨识技术路线

## 1. 数据预处理流程

### 1.1 数据加载与完整性检查
读取标准格式CSV文件，检查必要变量的完整性。支持两种动力数据格式：`delta_t`（油门百分比）或`rpm`（转速）。

**必要变量清单**：
- 状态量：`u_g, v_g, w_g, p, q, r, phi, theta, psi, lat, lon, h`
- 加速度：`ax, ay, az`（IMU比力数据）
- 控制量：`delta_e, delta_a, delta_r, delta_t/rpm`

**缺失值处理**：
使用线性插值修复，最大允许间隔0.1s（5帧@50Hz）。超出范围则拒绝数据。

### 1.2 坐标转换（LLA → NED）
选取首帧作为参考点 $(lat_0, lon_0, h_0)$，将经纬度转换为北东地坐标系：

$$
\begin{aligned}
x_{North} &= R \cdot (lat_{rad} - lat_{0,rad}) \\
y_{East} &= R \cdot \cos(lat_{0,rad}) \cdot (lon_{rad} - lon_{0,rad}) \\
z_{Down} &= -(h - h_0)
\end{aligned}
$$

其中 $R = 6371000$ m（地球半径）。

### 1.3 传感器噪声注入（可选）
为了使仿真数据更接近真实飞行记录，可在 `preprocess.py` 中启用噪声注入功能。仅对IMU数据添加噪声，姿态角和地速由导航滤波器输出，精度远高于IMU原始数据，不加噪声。

**噪声频率特性设计**：

- **IMU数据**（加速度、陀螺仪）：10-20Hz带通噪声
  - 模拟高频振动噪声（发动机/螺旋桨引起）
  - 真实IMU高频噪声（40-50Hz+）超出采样率表示能力，10-20Hz是可表示范围内的高频段

**噪声标准差配置**（针对420kg级固定翼无人机）：
- **IMU加速度** (`ax, ay, az`)：$\sigma = 0.1$ m/s²
- **IMU陀螺仪** (`p, q, r`)：$\sigma = 0.01$ rad/s（约0.57°/s）

**不加噪声的数据**：
- **姿态角** (`phi, theta, psi`)：飞控EKF融合输出，噪声频段（0.5-2Hz）与横向动力学频段重叠，加噪会破坏训练标签的物理自洽性
- **地速** (`u_g, v_g, w_g`)：GPS+卡尔曼滤波输出，同理
- **控制量**（`delta_e, delta_a, delta_r, delta_t`）：保持输入确定性
- **气动数据**（`TAS, alpha, beta, Q`）和高度（`h`）

**注意事项**：
- 使用固定随机种子（默认42）确保数据集可重复性
- 通过 `add_noise` 参数控制是否启用噪声注入
- 带通噪声通过白噪声 + 带通滤波器生成

### 1.4 低通滤波
对IMU原始测量数据进行零相位低通滤波：

**滤波器配置**：
- **IMU数据组** (`ax, ay, az, p, q, r`)：
  - 截止频率：8 Hz
  - 滤波器：2阶Butterworth
  - 原因：包含高频振动噪声（10-20Hz），且需要进行数值微分

**不滤波的数据**：
- **姿态角** (`phi, theta, psi`)：未注入噪声，无需滤波
- **地速** (`u_g, v_g, w_g`)：未注入噪声，无需滤波

**实现方法**：
使用 `scipy.signal.filtfilt` 实现零相位滤波，避免相位延迟。

### 1.5 状态变量重构
计算机体轴空速分量 $[u, v, w]^T$。根据仿真环境选择计算方式：

**无风模式**（几何投影法）：
利用地速向量与欧拉角构建旋转矩阵 $R_{nb}^T$，将NED系地速投影到机体轴：

$$
\begin{bmatrix} u \\ v \\ w \end{bmatrix} = R_{nb}^T(\phi, \theta, \psi) \begin{bmatrix} V_N \\ V_E \\ V_D \end{bmatrix}
$$

**有风模式**（气动定义法）：
利用真空速TAS与气流角分解：

$$
\begin{aligned}
u &= TAS \cdot \cos(\alpha) \cdot \cos(\beta) \\
v &= TAS \cdot \sin(\beta) \\
w &= TAS \cdot \sin(\alpha) \cdot \cos(\beta)
\end{aligned}
$$

### 1.6 数值微分与导数监督
**角加速度计算**：
使用Savitzky-Golay滤波器计算角加速度 $\dot{p}, \dot{q}, \dot{r}$：

```python
dot_p = savgol_filter(p, window_length=11, polyorder=2, deriv=1, delta=dt)
```

**优势**：
- 在微分的同时进行平滑，有效抑制噪声放大
- 对噪声数据的微分效果优于简单的中心差分
- 窗口长度11帧（0.22s @ 50Hz），多项式阶数2

**IMU杆臂效应修正**：
将IMU测量的加速度修正到重心（CG）位置：

$$
\mathbf{a}_{CG} = \mathbf{a}_{IMU} - \dot{\boldsymbol{\omega}} \times \mathbf{r}_{arm} - \boldsymbol{\omega} \times (\boldsymbol{\omega} \times \mathbf{r}_{arm})
$$

其中 $\mathbf{r}_{arm}$ 为IMU相对于CG的位置矢量。

### 1.7 数据归一化
采用混合归一化策略，针对不同物理量选择合适的方法：

**控制量**（Min-Max归一化）：
- 舵面 `delta_e, delta_a, delta_r`：归一化至 $[-1, 1]$
- 油门 `delta_t`：归一化至 $[0, 1]$

**状态量**（Z-Score标准化）：
- 速度、角速度、姿态角：$x_{norm} = \frac{x - \mu}{\sigma}$

**位置**（固定缩放）：
- NED坐标 `x, y, z`：除以1000转换为km级

归一化参数保存至 `scaler.pkl`，供训练和测试阶段复用。

### 1.8 轨迹切片
使用滑动窗口将连续飞行数据切割成固定长度片段：

**参数设置**：
- 窗口长度：100帧（2s @ 50Hz）
- 滑动步长：20帧（0.4s）
- 连续性检查：时间跨度误差 < 1e-4 × 窗口长度

**输出数据结构**：
```python
dataset = {
    'input_ode': (N, 12),      # 归一化初值 [u,v,w,p,q,r,φ,θ,ψ,x,y,z]
    'input_nn': (N, T, 4),     # 归一化控制序列 [δe,δa,δr,δt]
    'labels': {
        'traj': (N, T, 12),    # 物理状态轨迹真值
        'force': (N, T, 6)     # 物理力/导数真值 [ax,ay,az,ṗ,q̇,ṙ]
    },
    'states_norm': (N, T, 12)  # 归一化状态序列（用于监督）
}
```

### 1.9 注意事项
1. **psi角跳变问题**：使用 `np.unwrap()` 处理360°跳变。
2. **噪声注入与滤波**：
   - 噪声注入在 `preprocess.py` 中完成，通过 `add_noise` 参数控制
   - 低通滤波在 `data_processing.py` 中完成，采用分级策略
   - 角加速度微分使用 `savgol_filter`，在微分的同时进行平滑
3. **数据物理一致性**：训练前必须运行 `check_physics.py` 验证数据合理性。

---

## 2. 神经网络与物理模型

### 2.1 气动系数预测网络（CoefficientNet）
**架构设计**：
```
Input(10) → Linear(64) → Tanh 
          → Linear(128) → Tanh 
          → Linear(64) → Tanh 
          → Linear(6) → Output
```

**输入**（10维）：
$$
\mathbf{x}_{nn} = [u, v, w, p, q, r, \delta_e, \delta_a, \delta_r, \delta_t]^T
$$

**输出**（6维气动系数）：
$$
\mathbf{C}_{aero} = [C_X, C_Y, C_Z, C_l, C_m, C_n]^T
$$

**初始化策略**：
- 隐藏层：Xavier初始化（适配Tanh激活）
- 输出层：权重缩放0.01倍，偏置置零（避免初期预测过大）

**设计要点**：
- 不使用LSTM（时序信息由ODE积分器处理）
- 不使用ReLU（避免死神经元，Tanh提供更平滑的梯度）
- 推力不由神经网络预测，而是通过物理模型直接计算

### 2.2 物理系统ODE模块（AerialSystemODE）

#### 2.2.1 大气模型
基于ISA标准大气模型计算密度：

$$
\begin{aligned}
Alt &= h_0 - z_{NED} \\
T &= T_0 - L \cdot Alt \\
\rho &= \rho_0 \left(\frac{T}{T_0}\right)^{\frac{g}{RL} - 1}
\end{aligned}
$$

常数：$T_0=288.15$ K，$L=0.0065$ K/m，$\rho_0=1.225$ kg/m³，$g=9.80665$ m/s²。

#### 2.2.2 推力模型
采用Simulink三点插值模型（基于油门指令）：

$$
T(\delta_t) = \begin{cases}
12.275 \cdot \delta_t & \delta_t \leq 80 \\
982 + 30.2 \cdot (\delta_t - 80) & \delta_t > 80
\end{cases}
$$

推力偏心距修正：
$$
\mathbf{M}_T = \mathbf{r}_{offset} \times \mathbf{F}_T
$$

其中 $\mathbf{r}_{offset} = [0, 0, -0.75]^T$ m。

#### 2.2.3 气动力与力矩
动压计算：
$$
Q = \frac{1}{2} \rho V^2, \quad V = \sqrt{u^2 + v^2 + w^2}
$$

气动力与力矩：
$$
\begin{aligned}
\mathbf{F}_{aero} &= Q \cdot S \cdot [C_X, C_Y, C_Z]^T \\
\mathbf{M}_{aero} &= Q \cdot S \cdot [b \cdot C_l, \bar{c} \cdot C_m, b \cdot C_n]^T
\end{aligned}
$$

其中 $S=18.825$ m²（参考面积），$b=9.804$ m（翼展），$\bar{c}=1.932$ m（平均气动弦长）。

#### 2.2.4 重力投影
将重力从NED系转换到机体轴：

$$
\mathbf{F}_g = mg \begin{bmatrix} -\sin\theta \\ \cos\theta \sin\phi \\ \cos\theta \cos\phi \end{bmatrix}
$$

#### 2.2.5 六自由度刚体动力学方程
**线运动方程**：
$$
\dot{\mathbf{v}} = \frac{\mathbf{F}_{aero} + \mathbf{F}_T + \mathbf{F}_g}{m} - \boldsymbol{\omega} \times \mathbf{v}
$$

**角运动方程**（欧拉方程）：
$$
\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1} \left(\mathbf{M}_{aero} + \mathbf{M}_T - \boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega})\right)
$$

**运动学方程**：
姿态角导数（欧拉角微分方程）：
$$
\begin{aligned}
\dot{\phi} &= p + (q \sin\phi + r \cos\phi) \tan\theta \\
\dot{\theta} &= q \cos\phi - r \sin\phi \\
\dot{\psi} &= \frac{q \sin\phi + r \cos\phi}{\cos\theta}
\end{aligned}
$$

位置导数（机体速度转NED地速）：
$$
\begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{z} \end{bmatrix} = R_{nb}(\phi, \theta, \psi) \begin{bmatrix} u \\ v \\ w \end{bmatrix}
$$

#### 2.2.6 ODE标准接口
实现 `forward(t, state_norm)` 方法供 `torchdiffeq.odeint` 调用：

```python
def forward(self, t, state_norm):
    # 1. 获取当前时刻控制量（线性插值）
    u_norm = self.get_control_at_t(t)
    
    # 2. 神经网络预测气动系数
    nn_input = torch.cat([state_norm[:, :6], u_norm], dim=1)
    aero_coeffs = self.net(nn_input)
    
    # 3. 反归一化
    state_real = self.scaler.inverse_transform_vector(state_norm, self.state_keys)
    u_real = self.scaler.inverse_transform_vector(u_norm, self.control_keys)
    
    # 4. 代入物理方程
    dx_dt_real, _ = self.physics_equations(state_real, u_real, aero_coeffs)
    
    # 5. 导数归一化
    dx_dt_norm = self.scaler.scale_derivative_vector(dx_dt_real, self.state_keys)
    
    return torch.clamp(dx_dt_norm, min=-1000.0, max=1000.0)
```

**关键设计**：
- 控制量通过线性插值获取，支持任意时刻 $t$ 的查询
- 归一化空间积分，物理空间计算（数值稳定性）
- 导数截断防止数值爆炸

---

## 3. 正则化技术

### 3.1 Jacobian正则化（纵横向解耦）
基于经典气动力学的纵横向解耦假设，惩罚交叉耦合导数。

**物理依据**：
- 纵向系数 $[C_X, C_Z, C_m]$ 应主要依赖纵向状态 $[u, w, q]$
- 横向系数 $[C_Y, C_l, C_n]$ 应主要依赖横向状态 $[v, p, r]$

**实现方法**：
使用Hutchinson探针法高效估计掩码雅可比范数：

$$
\mathcal{L}_{jac} = \mathbb{E}_{\mathbf{v}_a} \left[ \left\| \frac{\partial \mathbf{C}_{long}}{\partial \mathbf{x}_{lat}} \mathbf{v}_a \right\|^2 \right] + \mathbb{E}_{\mathbf{v}_b} \left[ \left\| \frac{\partial \mathbf{C}_{lat}}{\partial \mathbf{x}_{long}} \mathbf{v}_b \right\|^2 \right]
$$

其中探针 $\mathbf{v}_a, \mathbf{v}_b \sim \mathcal{N}(0, \mathbf{I})$，使用2次探针估计。

**效果**：
解决了 $a_z$ 项预测稳态误差问题（2026-4-9更新）。

### 3.2 Hessian正则化（曲率惩罚）
惩罚网络输出对角速度的二阶导数，抑制预测曲线的非物理尖峰。

**惩罚维度**：
仅惩罚角速度 $[p, q, r]$ 的曲率，保护线速度 $[u, v, w]$ 和油门 $\delta_t$（承载诱导阻力、失速等非线性）。

**实现方法**：
使用Hessian-Vector Product (HVP) 估计：

$$
\mathcal{L}_{hes} = \frac{1}{n_{probes}} \sum_{i=1}^{n_{probes}} \left\| \mathbf{H} \mathbf{v}_i \right\|^2
$$

其中 $\mathbf{H}$ 为Hessian矩阵，$\mathbf{v}_i$ 为输入空间探针（仅在惩罚维度采样）。

**效果**：
解决了 $a_x$ 项预测稳态误差问题（2026-4-13更新）。

**已知局限**：
会顺带惩罚保护组与惩罚组的交叉二阶导（如 $\frac{\partial^2 C_m}{\partial w \partial \delta_e}$），在常规飞行包线内安全。

---

## 4. 训练策略

### 4.1 混合损失函数
将轨迹损失拆分为速度损失与运动学损失，力损失拆分为纵向与横航向，便于精细调整权重：

$$
\mathcal{L}_{total} = w_{vel} \mathcal{L}_{vel} + w_{kin} \mathcal{L}_{kin} + w_{f,lon} \mathcal{L}_{f,lon} + w_{f,lat} \mathcal{L}_{f,lat} + w_{jac} \mathcal{L}_{jac} + w_{hes} \mathcal{L}_{hes}
$$

**各项定义**：

1. **速度损失**（归一化空间MSE）：
$$
\mathcal{L}_{vel} = \text{MSE}(\mathbf{x}_{pred}[:, :6], \mathbf{x}_{gt}[:, :6])
$$
包含 $[u, v, w, p, q, r]$。

2. **运动学损失**（归一化空间MSE）：
$$
\mathcal{L}_{kin} = \text{MSE}(\mathbf{x}_{pred}[:, 6:], \mathbf{x}_{gt}[:, 6:])
$$
包含 $[\phi, \theta, \psi, x, y, z]$。

3. **纵向动力学损失**（方差归一化）：
$$
\mathcal{L}_{f,lon} = \frac{1}{3} \sum_{i \in \{a_x, a_z, \dot{q}\}} \frac{\text{MSE}(\mathbf{f}_{pred}^{(i)}, \mathbf{f}_{gt}^{(i)})}{\text{Var}(\mathbf{f}_{gt}^{(i)})}
$$

4. **横航向动力学损失**（方差归一化）：
$$
\mathcal{L}_{f,lat} = \frac{1}{3} \sum_{i \in \{a_y, \dot{p}, \dot{r}\}} \frac{\text{MSE}(\mathbf{f}_{pred}^{(i)}, \mathbf{f}_{gt}^{(i)})}{\text{Var}(\mathbf{f}_{gt}^{(i)})}
$$

**设计原理**：
- 纵横向力损失分离后，ReLoBRaLo可独立调节横向力的权重，避免横向梯度信号被纵向通道稀释
- 动力学损失使用真实状态（GT）输入模型计算预测力，而非ODE积分轨迹
- 比力 $\mathbf{a}$ 为非惯性系加速度（已扣除重力）

### 4.2 ReLoBRaLo动态多目标损失平衡
基于损失值历史衰减率的自适应权重调整算法。

**核心思想**：
- 长期分量：与初始损失 $\mathbf{L}_0$ 比较，评估整体进度
- 短期分量：与上一步损失 $\mathbf{L}_{t-1}$ 比较，评估最近进度
- Saudade 机制：通过伯努利随机变量偶尔触发"回溯冲击"

**权重更新公式**：

步骤一：计算相对平衡权重
$$
\lambda_i^{bal}(t, t') = m \cdot \frac{\exp(\frac{\mathcal{L}_i(t)}{\mathcal{T}\mathcal{L}_i(t')})}{\sum_{j=1}^m \exp(\frac{\mathcal{L}_j(t)}{\mathcal{T}\mathcal{L}_j(t')})}
$$

步骤二：Saudade 随机回溯
$$
\lambda_i^{hist}(t) = \rho \lambda_i(t-1) + (1-\rho) \lambda_i^{bal}(t, 0)
$$
其中 $\rho \sim \text{Bernoulli}(\mathbb{E}[\rho])$，期望值通常接近 1（如 0.999）

步骤三：指数衰减更新
$$
\lambda_i(t) = \alpha \lambda_i^{hist}(t) + (1-\alpha) \lambda_i^{bal}(t, t-1)
$$

**最终权重**：
$$
\mathbf{w}_{final} = \mathbf{w}_{base} \odot \boldsymbol{\lambda}(t)
$$

其中 $\mathbf{w}_{base}$ 为基础缩放因子（补偿量级差异），$\odot$ 为逐元素乘法。

**超参数**：
- $\alpha = 0.95$：指数衰减率（控制"记住过去"的能力）
- $\mathbb{E}[\rho] = 0.999$：Saudade 期望值（控制回溯频率，约每 1000 步触发一次）
- $\mathcal{T} = 1.0$：Softmax 温度（越小分布越尖锐）
- $\mathbf{w}_{base} = [1.0, 1.0, 1.0]$：速度、运动学、动力学基础权重

**正则化权重**（固定）：
- $w_{jac} = 10.0$
- $w_{hes} = 2.0$

### 4.3 训练流程

#### 4.3.1 预训练阶段（pretrain.py）
纯气动预训练，初始化神经网络参数。使用简化损失函数，仅监督气动系数预测能力。

#### 4.3.2 主训练阶段（train.py）
**优化器**：Adam，初始学习率 $1 \times 10^{-3}$

**学习率调度**：余弦退火（Cosine Annealing）
$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)
$$
其中 $\eta_{max} = 1 \times 10^{-3}$，$\eta_{min} = 1 \times 10^{-5}$，$T_{max} = 200$ epochs。

**梯度裁剪**：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**ODE积分**：
- 方法：RK4（4阶龙格-库塔）
- 步长：自适应（由torchdiffeq自动控制）
- 时间跨度：2s（100帧 @ 50Hz）

**训练循环伪代码**：
```python
for epoch in range(epochs):
    for batch in dataloader:
        # 1. 注入控制上下文
        model.set_control_context(t_span, controls)
        
        # 2. ODE积分
        pred_traj_norm = odeint(model, x0, t_span, method='rk4')
        
        # 3. 力/力矩诊断
        force_dict = model.predict_forces_and_moments(gt_states_norm, controls)
        
        # 4. 计算损失
        loss_vel, loss_kin, loss_force = loss_fn(pred_traj_norm, gt_traj_norm, 
                                                   pred_force, gt_force)
        loss_jac = model.net.compute_jacobian_regularization(gt_states_norm, controls)
        loss_hes = model.net.compute_hessian_regularization(gt_states_norm, controls)
        
        # 5. 加权求和
        w_vel, w_kin, w_force = relo.get_weights()
        loss = w_vel*loss_vel + w_kin*loss_kin + w_force*loss_force \
               + w_jac*loss_jac + w_hes*loss_hes
        
        # 6. 反向传播
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # 7. 更新ReLoBRaLo权重
    relo.update([avg_vel, avg_kin, avg_force])
    scheduler.step()
```

---

## 5. 关键技术要点总结

### 5.1 数据流设计
1. **归一化空间积分**：ODE在归一化空间求解，提升数值稳定性
2. **物理空间计算**：力/力矩计算在真实物理空间进行，保证物理一致性
3. **双重监督**：轨迹监督（ODE积分结果）+ 力监督（物理方程诊断）

### 5.2 模型架构特点
1. **混合驱动**：神经网络学习气动系数，物理方程提供结构约束
2. **可解释性**：气动系数具有明确物理意义，便于分析和验证
3. **泛化能力**：物理先验减少对数据的依赖，提升外推能力

### 5.3 正则化策略
1. **Jacobian正则化**：强制纵横向解耦，符合经典气动理论
2. **Hessian正则化**：抑制角速度曲率，消除非物理振荡
3. **掩码机制**：精细控制惩罚范围，保护必要的非线性

### 5.4 训练技巧
1. **两阶段训练**：预训练 → 主训练，加速收敛
2. **动态权重平衡**：ReLoBRaLo自适应调整多目标权重
3. **梯度裁剪**：防止梯度爆炸，稳定训练过程
4. **余弦退火**：学习率平滑衰减，避免震荡

### 5.5 已解决的关键问题
1. **psi角跳变**：使用 `np.unwrap()` 处理（2026-1-8）
2. **数值发散**：导数截断 + 梯度裁剪（2026-3-10）
3. **$a_z$ 稳态误差**：Jacobian正则化（2026-4-9）
4. **$a_x$ 稳态误差**：Hessian正则化（2026-4-13）
5. **噪声数据处理**：分级滤波策略 + Savgol微分（2026-4-29）

---

## 6. 物理参数配置

### 6.1 飞行器参数
```python
props = {
    'm': 420,                   # 质量 (kg)
    'S': 18.825,                # 参考面积 (m²)
    'b': 9.804,                 # 翼展 (m)
    'c': 1.932,                 # 平均气动弦长 (m)
    'h0': 1100.5,               # 起飞高度 (m)
    'T_offset': [0, 0, -0.75],  # 推力偏心距 (m)
    'I': [                      # 惯量矩阵 (kg·m²)
        [ 539.246,      0.0, -105.374],
        [     0.0, 694.0101,      0.0],
        [-105.374,      0.0, 1018.916]
    ]
}
```

### 6.2 训练超参数
```python
config = {
    'batch_size': 64,
    'max_lr': 1e-3,
    'min_lr': 1e-5,
    'epochs': 200,
    'window_size': 100,         # 2s @ 50Hz
    'stride': 20,               # 0.4s
    'dt': 0.02,                 # 50Hz
}
```

### 6.3 损失函数权重
```python
loss_config = {
    'alpha': 0.95,              # ReLoBRaLo 指数衰减率（控制"记住过去"的能力）
    'rho': 0.999,               # ReLoBRaLo Saudade 期望值（控制回溯频率，约每1000步触发一次）
    'temperature': 1.0,         # Softmax 温度（越小分布越尖锐）
    'base_weights': [1.0, 1.0, 1.0, 1.0],  # [vel, kin, force_lon, force_lat] 基础缩放因子
    'w_jac': 10.0,              # Jacobian 正则化固定权重
    'w_hes': 2.0,               # Hessian 正则化固定权重
}
```

---

## 7. 文件结构与运行流程

### 7.1 核心文件
- `preprocess.py`：坐标系转换（定制系 → 标准NED系）
- `data_processing.py`：数据预处理流水线
- `flight_scaler.py`：归一化工具类
- `check_physics.py`：物理一致性校验
- `NeuralODEFunc.py`：神经网络 + 物理模型 + ODE接口
- `relobralo.py`：动态多目标损失平衡算法
- `pretrain.py`：纯气动预训练
- `train.py`：主训练程序
- `test.py`：测试与可视化

### 7.2 运行流程
```bash
# 1. 坐标系转换
python preprocess.py

# 2. 物理一致性校验
python check_physics.py

# 3. 数据预处理（生成dataset.pt和scaler.pkl）
python data_processing.py

# 4. 预训练（生成pretrained_coeffs.pth）
python pretrain.py

# 5. 主训练（生成model_weights.pth）
python train.py

# 6. 测试
python test.py
```

---

## 8. 技术创新点

1. **混合驱动架构**：神经网络 + 物理方程，兼顾拟合能力与可解释性
2. **掩码正则化**：Jacobian/Hessian正则化引入掩码矩阵，精细控制惩罚范围
3. **ReLoBRaLo算法**：基于历史衰减率的自适应多目标权重平衡
4. **双重监督机制**：轨迹监督 + 力/力矩监督，提升物理一致性
5. **归一化空间积分**：在归一化空间求解ODE，提升数值稳定性

---

**文档版本**：v1.0  
**最后更新**：2026-04-20  
**适用代码版本**：2026-04-18更新后
