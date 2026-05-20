import torch
import torch.nn.functional as F


class ReLoBRaLo:
    """
    ReLoBRaLo: Relative Loss Balancing with Random Lookback

    严格按照文献实现的动态多目标损失平衡算法。

    算法核心三步骤：
    1. 计算相对平衡权重 λ^bal(t, t')
    2. Saudade 随机回溯：λ^hist(t) = ρ * λ(t-1) + (1-ρ) * λ^bal(t, 0)
    3. 指数衰减更新：λ(t) = α * λ^hist + (1-α) * λ^bal(t, t-1)

    Args:
        m (int): 损失项数量
        alpha (float): 指数衰减率，控制"记住过去"的能力 (推荐: 0.95)
        rho (float): Saudade 伯努利分布的期望值，控制回溯频率 (推荐: 0.999)
        temperature (float): Softmax 温度参数 (推荐: 0.5-2.0)
        base_weights (list): 各损失项的基础缩放因子，用于补偿量级差异
        max_epochs (int): 最大训练轮数 (用于预分配历史缓冲区)
    """
    def __init__(self, m=3, alpha=0.95, rho=0.999, temperature=1.0,
                 base_weights=None, max_epochs=200):
        self.m = m  # 损失项数量
        self.alpha = alpha  # 指数衰减率
        self.rho = rho  # Saudade 期望值
        self.temperature = temperature  # 温度参数

        # 基础权重（工程改进，用于补偿量级差异）
        self.base_weights = torch.tensor(
            base_weights if base_weights is not None else [1.0] * m,
            dtype=torch.float32
        )

        # 历史记录
        self.loss_history = torch.zeros(max_epochs, m)  # 损失历史
        self.L0 = None  # 初始损失 L(0)
        self.lambda_prev = torch.ones(m)  # 上一步的权重 λ(t-1)
        self.current_epoch = 0

        # 权重历史（用于可视化）
        self.weight_history = []

    def _compute_lambda_bal(self, L_current, L_reference):
        """
        计算相对平衡权重 λ^bal(t, t')

        公式: λ_i^bal = m * exp(L_i(t) / (T * L_i(t'))) / Σ exp(L_j(t) / (T * L_j(t')))

        Args:
            L_current: 当前损失 L(t)
            L_reference: 参考损失 L(t')

        Returns:
            λ^bal: 相对平衡权重，形状 (m,)
        """
        # 计算比值 L(t) / L(t')
        ratios = L_current / (L_reference + 1e-8)

        # Softmax 归一化，乘以 m 保证权重和为 m
        lambda_bal = F.softmax(ratios / self.temperature, dim=0) * self.m

        return lambda_bal

    def update(self, epoch_losses):
        """
        每个 epoch 末调用，更新权重。

        Args:
            epoch_losses: 当前 epoch 的平均损失，列表或张量，形状 (m,)

        Returns:
            final_weights: 最终权重（包含 base_weights），形状 (m,)
        """
        L_current = torch.tensor(epoch_losses, dtype=torch.float32)
        t = self.current_epoch

        # 记录当前损失
        self.loss_history[t] = L_current

        # 第一个 epoch：初始化
        if t == 0:
            self.L0 = L_current.clone()
            self.lambda_prev = torch.ones(self.m)
            lambda_current = torch.ones(self.m)
        else:
            # 步骤1：计算相对平衡权重
            # λ^bal(t, 0): 与初始损失比较（长期进度）
            lambda_bal_long = self._compute_lambda_bal(L_current, self.L0)

            # λ^bal(t, t-1): 与上一步损失比较（短期进度）
            L_prev = self.loss_history[t - 1]
            lambda_bal_short = self._compute_lambda_bal(L_current, L_prev)

            # 步骤2：Saudade 随机回溯
            # ρ ~ Bernoulli(self.rho)
            rho_sample = torch.bernoulli(torch.tensor(self.rho)).item()
            lambda_hist = rho_sample * self.lambda_prev + (1 - rho_sample) * lambda_bal_long

            # 步骤3：指数衰减更新
            # λ(t) = α * λ^hist + (1-α) * λ^bal(t, t-1)
            lambda_current = self.alpha * lambda_hist + (1 - self.alpha) * lambda_bal_short

        # 更新历史
        self.lambda_prev = lambda_current.clone()

        # 最终权重 = 基础权重 * 动态权重
        final_weights = self.base_weights * lambda_current

        # 记录权重历史
        self.weight_history.append(final_weights.clone().tolist())

        self.current_epoch += 1

        return final_weights.clone()

    def get_weights(self):
        """
        返回当前最终权重（包含 base_weights）

        Returns:
            tuple: 各损失项的权重
        """
        final_weights = self.base_weights * self.lambda_prev
        return tuple(final_weights[i].item() for i in range(self.m))

    def get_dynamic_weights(self):
        """
        返回当前动态权重（不含 base_weights）

        Returns:
            tuple: 各损失项的动态权重 λ(t)
        """
        return tuple(self.lambda_prev[i].item() for i in range(self.m))
