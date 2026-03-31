import pandas as pd
import numpy as np

def preprocess_flight_data(input_csv, output_csv):
    """
    将定制坐标系(北天东/前上右)的飞行数据转换为标准航空坐标系(北东地/前右下)
    """
    print(f"正在读取原始数据: {input_csv} ...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_csv}")
        return

    t_start = 150.0
    t_end = 250.0
        
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
    df_target['p'] = df['Wx']          # 滚转率不变
    df_target['q'] = df['Wz']          # 原Z轴(右)对应的就是俯仰率
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
    
    df_target.to_csv(output_csv, index=False)
    print(f"数据转换成功！已输出标准坐标系数据至: {output_csv}")
    print(f"包含变量: {list(df_target.columns)}")

if __name__ == "__main__":
    INPUT_FILE = "Document.csv"
    OUTPUT_FILE = "Document0.csv"
    preprocess_flight_data(INPUT_FILE, OUTPUT_FILE)