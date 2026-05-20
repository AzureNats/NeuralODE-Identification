import pandas as pd
import numpy as np
import scipy.signal

def generate_bandpass_noise(length, dt, sigma, freq_low, freq_high, order=4, seed=None):
    """
    生成带通噪声

    Args:
        length (int): 噪声序列长度
        dt (float): 采样时间间隔
        sigma (float): 目标标准差
        freq_low (float): 带通下限频率 (Hz)
        freq_high (float): 带通上限频率 (Hz)
        order (int): 滤波器阶数
        seed (int): 随机种子

    Returns:
        np.ndarray: 带通噪声序列
    """
    if seed is not None:
        np.random.seed(seed)

    # 生成白噪声
    white_noise = np.random.normal(0, 1, length)

    # 设计带通滤波器
    nyquist = 0.5 / dt
    low = freq_low / nyquist
    high = freq_high / nyquist

    # 确保频率在有效范围内
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))

    b, a = scipy.signal.butter(order, [low, high], btype='band')

    # 滤波得到带通噪声
    bandpass_noise = scipy.signal.filtfilt(b, a, white_noise)

    # 重新缩放到目标标准差
    if np.std(bandpass_noise) > 1e-10:
        bandpass_noise = bandpass_noise * (sigma / np.std(bandpass_noise))
    else:
        bandpass_noise = np.zeros(length)

    return bandpass_noise


def preprocess_flight_data(input_csv, output_csv, add_noise=False, noise_seed=42):
    """
    将定制坐标系(北天东/前上右)的飞行数据转换为标准航空坐标系(北东地/前右下)

    Args:
        input_csv (str): 输入CSV文件路径
        output_csv (str): 输出CSV文件路径
        add_noise (bool): 是否注入传感器噪声，默认False
        noise_seed (int): 随机种子，默认42
    """
    print(f"正在读取原始数据: {input_csv} ...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_csv}")
        return

    t_start = 60.0
    t_end = 500.0
    
    df = df[(df['Time(s)'] >= t_start) & (df['Time(s)'] <= t_end)]
    df = df.reset_index(drop=True)
    
    original_start = df['Time(s)'].iloc[0]
    df['Time(s)'] = df['Time(s)'] - original_start
    df['Time(s)'] = df['Time(s)'].round(3)

    df_target = pd.DataFrame()

    # 1. 时间戳
    df_target['time'] = df['Time(s)']

    # # 2. 机体系线速度 (前-右-下)
    # df_target['u'] = df['Vxt']       # 前向不变
    # df_target['v'] = df['Vzt']       # 右向(原Z)变为Y
    # df_target['w'] = -df['Vyt']      # 上向(原Y)取反变下向

    # 3. 机体系角速度 (滚转、俯仰、偏航)
    df_target['p'] = df['Wx']          # 滚转角速度不变
    df_target['q'] = df['Wz']          # 原Z轴(右)对应的就是俯仰角速度
    df_target['r'] = df['Wy']          # 原Y轴(上)的偏航为左偏，取反变右偏

    # 4. 欧拉角 (Roll, Pitch, Yaw)
    df_target['phi'] = df['gama']      # 滚转角
    df_target['theta'] = df['theta']   # 俯仰角
    df_target['psi'] = df['psi']       # 原偏航角为左偏，取反变为右偏航
    df_target['psi'] = np.unwrap(df_target['psi'].values, discont=np.pi)

    # 5. 地面导航系速度 (北-东-地)
    df_target['u_g'] = df['Vn']        # 北向不变
    df_target['v_g'] = df['Ve']        # 东向不变
    df_target['w_g'] = df['Vd']        # 原Vd实为天向(Up)，取反变为地向(Down)

    # 6. 机体系加速度 (前-右-下)
    # 6.1 获取并转换仿真器输出的纯运动学加速度
    ax_sim = df['Ax_body']        # 前向不变
    ay_sim = df['Az_body']        # 原Z(右)变为Y(右)
    az_sim = -df['Ay_body']       # 原Y(上)取反变为Z(下)

    # 6.2 计算重力在机体轴上的投影分量
    g = 9.80665
    phi = df['gama'].values       # 滚转角 (rad)
    theta = df['theta'].values    # 俯仰角 (rad)
    
    gx = -g * np.sin(theta)
    gy =  g * np.cos(theta) * np.sin(phi)
    gz =  g * np.cos(theta) * np.cos(phi)

    # 6.3 转换为真实的 IMU 比力数据 (f = a - g)
    df_target['ax'] = ax_sim - gx
    df_target['ay'] = ay_sim - gy
    df_target['az'] = az_sim - gz

    # 7. 地理位置与高度
    df_target['lat'] = df['lat_re']
    df_target['lon'] = df['lon_re']
    df_target['h'] = df['H_re']

    # 8. 控制量
    df_target['delta_t'] = df['Gain_dp']
    df_target['delta_e'] = df['Gain_de']
    df_target['delta_a'] = df['Gain_da']
    df_target['delta_r'] = df['Gain_dr']

    # 9. 气动数据 (不考虑地面)
    df_target['TAS'] = df['TAS']
    df_target['Q'] = df['q']
    df_target['alpha'] = df['alpha']
    df_target['beta'] = df['beta']
    df_target['FX'] = -df['FX']
    df_target['FY'] = df['FZ']
    df_target['FZ'] = -df['FY']
    df_target['L'] = df['Mx']
    df_target['M'] = df['Mz']
    df_target['N'] = -df['My']

    time_cents = np.round(df_target['time'].values * 100).astype(int)
    df_target = df_target[time_cents % 2 == 0].reset_index(drop=True)

    cols_to_clean = df_target.columns.drop('time')
    df_target[cols_to_clean] = df_target[cols_to_clean].mask(df_target[cols_to_clean].abs() < 1e-5, 0.0)

    # 传感器真实噪声注入模块 (带通噪声)
    if add_noise:
        print("正在为仿真数据注入带通噪声以模拟真实传感器...")

        # 采样时间间隔
        dt = df_target['time'].iloc[1] - df_target['time'].iloc[0]
        data_length = len(df_target)

        # 仅对IMU数据添加噪声 (高频振动噪声，10-20Hz)
        # 姿态角和地速由导航滤波器输出，精度远高于IMU原始数据，不加噪声
        noise_configs = {
            'IMU': {
                'cols': ['ax', 'ay', 'az', 'p', 'q', 'r'],
                'sigma': {'ax': 0.1, 'ay': 0.1, 'az': 0.1,
                         'p': 0.01, 'q': 0.01, 'r': 0.01},
                'freq_low': 10.0,
                'freq_high': 20.0
            }
        }

        # 对每组数据注入对应频段的噪声
        for group_name, config in noise_configs.items():
            freq_low = config['freq_low']
            freq_high = config['freq_high']

            for col in config['cols']:
                if col in df_target.columns:
                    sigma = config['sigma'][col]

                    # 生成带通噪声（每列使用不同的随机种子）
                    seed_offset = hash(col) % 10000
                    noise = generate_bandpass_noise(
                        data_length, dt, sigma,
                        freq_low, freq_high,
                        order=4,
                        seed=noise_seed + seed_offset
                    )

                    df_target[col] = df_target[col] + noise

            print(f"  -> {group_name}组: 已注入{freq_low}-{freq_high}Hz带通噪声")

        print(f"噪声注入完成 (随机种子: {noise_seed})。")

    df_target.to_csv(output_csv, index=False)
    print(f"数据转换成功！已输出标准坐标系数据至: {output_csv}")
    print(f"包含变量: {list(df_target.columns)}")

if __name__ == "__main__":
    INPUT_FILE = "Document5020.csv"
    OUTPUT_FILE = "Document52.csv"
    ADD_NOISE = True
    NOISE_SEED = 52
    preprocess_flight_data(INPUT_FILE, OUTPUT_FILE, add_noise=ADD_NOISE, noise_seed=NOISE_SEED)