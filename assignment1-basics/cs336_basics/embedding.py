import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from einops import rearrange, einsum


class Embedding(nn.Module):

    num_embeddings: int
    embedding_dim: int
    weight: Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        # Initialize weights: mean=0, std=1, truncated to [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long()
        return self.weight[token_ids]


class RotaryPositionalEmbedding(nn.Module):

    theta: float
    d_k: int
    max_seq_len: int
    sin: Tensor
    cos: Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 预计算角度参数
        indices = torch.arange(0, d_k//2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2 * indices / d_k)  # shape (d_k//2)

        # 构建位置和频率的笛卡尔积
        positions = torch.arange(max_seq_len, device=device)
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (max_seq_len, d_k//2)

        # 注册为非持久化缓冲区
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        # x & out (..., seq_len, d_k)
        # token_positions: same leading dimensions as x's up to seq_len (..., seq_len)

        # 提取预计算的三角值
        cos = self.cos[token_positions]  # (..., seq_len, d_half)
        sin = self.sin[token_positions]

        # 复数对重组 d_k -> (d_pair, two=2)
        x_paired = rearrange(x, '... t (d_pair two) -> ... t d_pair two', two=2)

        # 旋转矩阵隐式应用（Einops 广播）
        '''
        x_paired[..., 0]: [x0, x2]
        x_paired[..., 1]: [x1, x3]

              （原始形状）         （内存中对齐方式）
        cos = [c0, c1]       → ([c0, c1], ... , [c0, c1]) # 广播到与x_paired相同维度
        sin = [s0, s1]       → ([s0, s1], ... , [s0, s1])

        x_rotated_0 = x0*cos_k - x1*sin_k
        term1 = x_paired[..., 0] * cos  # shape (1, 2)
                = [x0*c0, x2*c1]
        term2 = x_paired[..., 1] * sin  # shape (1, 2)
                = [x1*s0, x3*s1]
        x_rotated_0 = term1 - term2
                    = [x0*c0 - x1*s0, x2*c1 - x3*s1]
        x_rotated = [
            [
                [[x0*c0 - x1*s0,  x0*s0 + x1*c0],  # 旋转后的第一个复数对
                [x2*c1 - x3*s1,  x2*s1 + x3*c1]]  # 旋转后的第二个复数对
            ]
        ]
        # shape=(1, 2, 2)
        '''
        x_rotated = torch.stack(
            (
                x_paired[..., 0] * cos - x_paired[..., 1] * sin,
                x_paired[..., 0] * sin + x_paired[..., 1] * cos
            ),
            dim=-1
        )

        # 恢复原始维度顺序
        return rearrange(x_rotated, '... t d_pair two -> ... t (d_pair two)')
