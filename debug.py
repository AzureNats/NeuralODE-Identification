"""
debug.py - 主训练 vs 预训练 Cz 退化诊断
对比预训练权重和主训练权重在同一数据上的 Cz/az 表现，
确认主训练阶段是否放大了预训练中本来很小的 Cz 误差。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from NeuralODEFunc import CoefficientNet, AerialSystemODE
from flight_scaler import FlightDataScaler
from pretrain import StaticAeroDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_net(weights_path):
    """加载 CoefficientNet 权重（兼容多种 state_dict 格式）"""
    net = CoefficientNet().to(device)
    state = torch.load(weights_path, map_location=device)

    # 提取所有包含网络权重的 key，统一映射到 CoefficientNet 的格式 (net.0.weight 等)
    # 可能的格式: 0.weight / net.0.weight / net.net.0.weight / net.net.net.0.weight ...
    mapped = {}
    for k, v in state.items():
        # 剥掉所有前缀，直到剩下 "数字.weight/bias" 的纯 Sequential key
        stripped = k
        while stripped.startswith('net.'):
            stripped = stripped[4:]
        # 只保留 Sequential 层的参数 (如 0.weight, 2.bias ...)
        if stripped and stripped[0].isdigit():
            mapped[f'net.{stripped}'] = v

    if not mapped:
        raise RuntimeError(f"无法从 state_dict 中提取网络权重。Keys: {list(state.keys())[:5]}")
    net.load_state_dict(mapped)

    net.eval()
    return net


def main():
    # ============ 配置 ============
    CSV_PATH = 'Document41.csv'
    SCALER_PATH = 'scaler41.pkl'
    PRETRAIN_WEIGHTS = 'pretrained_coeffs.pth'
    TRAINED_WEIGHTS  = 'model_weights.pth'
    SAVE_DIR = './debug_results'

    PROPS = {
        'm': 420,
        'S': 18.825,
        'b': 9.804,
        'c': 1.932,
        'h0': 1100.5,
        'T_offset': [0, 0, -0.75],
        'I': [
            [ 539.246,      0.0, -105.374],
            [     0.0, 694.0101,      0.0],
            [-105.374,      0.0, 1018.916]
        ]
    }

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ============ 1. 加载数据 ============
    dataset = StaticAeroDataset(CSV_PATH, SCALER_PATH, PROPS)
    all_inputs = dataset.nn_inputs     # (N, 10)
    all_gt     = dataset.gt_coeffs     # (N, 6)

    # ============ 2. 两组权重分别推理 ============
    net_pre = load_net(PRETRAIN_WEIGHTS)
    net_trn = load_net(TRAINED_WEIGHTS)

    with torch.no_grad():
        pred_pre = net_pre(all_inputs.to(device)).cpu().numpy()
        pred_trn = net_trn(all_inputs.to(device)).cpu().numpy()

    gt_np = all_gt.numpy()
    err_pre = pred_pre - gt_np
    err_trn = pred_trn - gt_np

    coeff_names = ['Cx', 'Cy', 'Cz', 'Cl', 'Cm', 'Cn']
    df = pd.read_csv(CSV_PATH)
    n = len(dataset)

    # ============ 3. 统计对比表 ============
    print("\n" + "=" * 85)
    print(f"{'系数':<6} {'| Pretrain RMSE':>16} {'Mean Err':>12} {'| Trained RMSE':>16} {'Mean Err':>12} {'| RMSE Ratio':>13}")
    print("-" * 85)
    for i, name in enumerate(coeff_names):
        rmse_p = np.sqrt(np.mean(err_pre[:, i]**2))
        rmse_t = np.sqrt(np.mean(err_trn[:, i]**2))
        ratio = rmse_t / (rmse_p + 1e-12)
        print(f"{name:<6} | {rmse_p:>14.6f} {err_pre[:, i].mean():>12.6f} "
              f"| {rmse_t:>14.6f} {err_trn[:, i].mean():>12.6f} | {ratio:>11.2f}x")
    print("=" * 85)

    # ============ 4. 六通道误差对比直方图 ============
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Error Distribution: Pretrained (blue) vs Trained (red)', fontsize=16)

    for i, name in enumerate(coeff_names):
        ax = axes[i // 3, i % 3]
        e_p = err_pre[:, i]
        e_t = err_trn[:, i]
        lo = min(e_p.min(), e_t.min())
        hi = max(e_p.max(), e_t.max())
        bins = np.linspace(lo, hi, 100)

        ax.hist(e_p, bins=bins, alpha=0.6, color='steelblue', label=f'Pretrain (std={e_p.std():.5f})')
        ax.hist(e_t, bins=bins, alpha=0.6, color='tomato',    label=f'Trained  (std={e_t.std():.5f})')
        ax.axvline(e_p.mean(), color='blue', linestyle='--', linewidth=1.5)
        ax.axvline(e_t.mean(), color='red',  linestyle='--', linewidth=1.5)
        ax.set_xlabel(f'{name} Error')
        ax.set_title(name)
        ax.legend(fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = os.path.join(SAVE_DIR, 'compare_error_hist.png')
    plt.savefig(path1, dpi=200)
    print(f"\n误差对比直方图: {path1}")

    # ============ 5. Cz 散点对比 (并排) ============
    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Cz: Prediction vs Ground Truth', fontsize=16)

    for ax, pred, label, color in [
        (ax_l, pred_pre[:, 2], 'Pretrained', 'steelblue'),
        (ax_r, pred_trn[:, 2], 'Trained',    'tomato')
    ]:
        gt_cz = gt_np[:, 2]
        rmse = np.sqrt(np.mean((pred - gt_cz)**2))
        ax.scatter(gt_cz, pred, s=1, alpha=0.3, color=color)
        lo = min(gt_cz.min(), pred.min())
        hi = max(gt_cz.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5)
        ax.set_xlabel('Cz (GT)')
        ax.set_ylabel('Cz (Pred)')
        ax.set_title(f'{label}  (RMSE={rmse:.5f})')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = os.path.join(SAVE_DIR, 'compare_cz_scatter.png')
    plt.savefig(path2, dpi=200)
    print(f"Cz 散点对比: {path2}")

    # ============ 6. Cz 时序对比 + az 等效误差 ============
    time = np.arange(n) * 0.02
    Q_arr = df['Q'].values[:n]
    S, m = PROPS['S'], PROPS['m']

    cz_err_pre = err_pre[:, 2]
    cz_err_trn = err_trn[:, 2]
    az_err_pre = Q_arr * S * cz_err_pre / m
    az_err_trn = Q_arr * S * cz_err_trn / m

    fig3, axes3 = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig3.suptitle('Cz Time-Series: Pretrained vs Trained', fontsize=16)

    # Row 1: Cz 真值 + 两组预测
    axes3[0].plot(time, gt_np[:, 2], 'k-', linewidth=1, label='GT')
    axes3[0].plot(time, pred_pre[:, 2], color='steelblue', linewidth=0.8, alpha=0.8, label='Pretrained')
    axes3[0].plot(time, pred_trn[:, 2], color='tomato', linewidth=0.8, alpha=0.8, label='Trained')
    axes3[0].set_ylabel('Cz')
    axes3[0].legend(loc='lower left')
    axes3[0].grid(True, linestyle=':', alpha=0.5)

    # Row 2: Cz 误差对比
    axes3[1].plot(time, cz_err_pre, color='steelblue', linewidth=0.8, label='Pretrain Err')
    axes3[1].plot(time, cz_err_trn, color='tomato', linewidth=0.8, label='Trained Err')
    axes3[1].axhline(0, color='grey', linestyle='--')
    axes3[1].set_ylabel('Cz Error')
    axes3[1].legend()
    axes3[1].grid(True, linestyle=':', alpha=0.5)

    # Row 3: 换算 az 误差
    axes3[2].plot(time, az_err_pre, color='steelblue', linewidth=0.8, label='Pretrain')
    axes3[2].plot(time, az_err_trn, color='tomato', linewidth=0.8, label='Trained')
    axes3[2].axhline(0, color='grey', linestyle='--')
    axes3[2].set_ylabel('Implied az Error (m/s²)')
    axes3[2].set_xlabel('Time (s)')
    axes3[2].legend()
    axes3[2].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path3 = os.path.join(SAVE_DIR, 'compare_cz_timeseries.png')
    plt.savefig(path3, dpi=200)
    print(f"Cz 时序对比: {path3}")

    # ============ 7. Cz 误差 vs alpha (两组叠加) ============
    alpha = df['alpha'].values[:n]

    fig4, (ax4l, ax4r) = plt.subplots(1, 2, figsize=(14, 5))
    fig4.suptitle('Cz Error vs Alpha: Pretrained vs Trained', fontsize=16)

    for ax, err, label, color in [
        (ax4l, cz_err_pre, 'Pretrained', 'steelblue'),
        (ax4r, cz_err_trn, 'Trained',    'tomato')
    ]:
        ax.scatter(alpha, err, s=1, alpha=0.3, color=color)
        ax.axhline(0, color='grey', linestyle='--')
        z = np.polyfit(alpha, err, 1)
        x_fit = np.linspace(alpha.min(), alpha.max(), 100)
        ax.plot(x_fit, np.polyval(z, x_fit), 'orange', linewidth=2,
                label=f'slope={z[0]:.4e}')
        ax.set_xlabel('alpha (rad)')
        ax.set_ylabel('Cz Error')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path4 = os.path.join(SAVE_DIR, 'compare_cz_vs_alpha.png')
    plt.savefig(path4, dpi=200)
    print(f"Cz vs alpha 对比: {path4}")

    # ============ 8. 汇总统计 ============
    print(f"\n--- az 等效误差汇总 ---")
    print(f"  Pretrained:  RMSE={np.sqrt(np.mean(az_err_pre**2)):.4f} m/s²,  "
          f"max |err|={np.abs(az_err_pre).max():.4f} m/s²")
    print(f"  Trained:     RMSE={np.sqrt(np.mean(az_err_trn**2)):.4f} m/s²,  "
          f"max |err|={np.abs(az_err_trn).max():.4f} m/s²")
    print(f"  放大倍数:    {np.sqrt(np.mean(az_err_trn**2)) / (np.sqrt(np.mean(az_err_pre**2)) + 1e-12):.1f}x")

    print(f"\n所有图片已保存至: {SAVE_DIR}/")


if __name__ == '__main__':
    main()
