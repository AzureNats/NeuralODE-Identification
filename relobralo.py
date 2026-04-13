import torch
import torch.nn.functional as F


class ReLoBRaLo:
    """
    ReLoBRaLo: Relative Loss Balancing with Random Lookback

    基于损失值历史衰减率的动态多目标损失平衡算法。
    按 epoch 粒度更新权重，每个 epoch 末调用 update()。

    支持 base_weights 参数，用于补偿各损失项之间的量级差异。
    最终输出权重 = base_weight * dynamic_weight。

    Args:
        n_losses (int): 损失项数量
        alpha (float): 长期 vs 短期平衡系数 (越大越偏长期)
        beta (float): EMA 平滑系数 (越大越平滑)
        tau (float): Softmax 温度 (越大分布越均匀)
        base_weights (list): 各损失项的基础缩放因子，用于补偿量级差异
        max_epochs (int): 最大训练轮数 (用于预分配历史缓冲区)
    """
    def __init__(self, n_losses=3, alpha=0.999, beta=0.999, tau=1.0,
                 base_weights=None, max_epochs=200):
        self.n = n_losses
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.base_weights = torch.tensor(
            base_weights if base_weights is not None else [1.0] * n_losses,
            dtype=torch.float32
        )

        self.loss_history = torch.zeros(max_epochs, n_losses)
        self.L0 = None
        self.dynamic_weights = torch.ones(n_losses)  # ReLoBRaLo 动态部分
        self.weight_history = []
        self.current_epoch = 0

    def _softmax_weights(self, loss_current, loss_ref):
        """
        计算 Softmax 归一化的相对权重
        w_j = softmax(L_j^curr / L_j^ref / tau) * n_losses
        """
        ratios = loss_current / (loss_ref + 1e-8)
        return F.softmax(ratios / self.tau, dim=0) * self.n

    def update(self, epoch_losses):
        """
        每个 epoch 末调用，输入该 epoch 的平均 loss，更新权重。

        Args:
            epoch_losses: [loss_traj, loss_force, loss_jac] 的列表或张量

        Returns:
            weights: 形状 (n_losses,) 的权重张量
        """
        losses = torch.tensor(epoch_losses, dtype=torch.float32)
        i = self.current_epoch
        self.loss_history[i] = losses

        if i == 0:
            self.L0 = losses.clone()
            self.dynamic_weights = torch.ones(self.n)
        else:
            # 长期分量: 与初始 loss 比较
            w_hat_long = self._softmax_weights(losses, self.L0)

            # 短期分量: 随机回溯，从历史中随机采样一个 epoch 比较
            lookback_idx = torch.randint(0, i, (1,)).item()
            L_lookback = self.loss_history[lookback_idx]
            w_hat_short = self._softmax_weights(losses, L_lookback)

            # 混合长期与短期
            w_combined = self.alpha * w_hat_long + (1 - self.alpha) * w_hat_short

            # EMA 平滑
            self.dynamic_weights = self.beta * self.dynamic_weights + (1 - self.beta) * w_combined

        # 最终权重 = 基础缩放 * 动态权重
        final_weights = self.base_weights * self.dynamic_weights
        self.weight_history.append(final_weights.clone().tolist())
        self.current_epoch += 1

        return final_weights.clone()

    def get_weights(self):
        """返回当前最终权重，元素数量与 n_losses 一致"""
        final = self.base_weights * self.dynamic_weights
        return tuple(final[i].item() for i in range(self.n))