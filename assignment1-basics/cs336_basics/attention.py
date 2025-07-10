import torch
import math
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.parameter import Parameter
import einops

from cs336_basics.embedding import RotaryPositionalEmbedding
from cs336_basics.linear import LinearNobias


def softmax(
    x: Tensor,
    d: int,
) -> Tensor:
    max_vals = torch.max(x, dim=d, keepdim=True).values
    shifted_x = x - max_vals  # Broadcast: subtract max of dim `d` along that dim
    exp_x = torch.exp(shifted_x)
    sum_exp = exp_x.sum(dim=d, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    scores = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (Q.shape[-1] ** 0.5)

    if mask is not None:
        scores = torch.where(mask, scores, -torch.inf)

    weights = softmax(scores, d=-1)
    output = einops.einsum(weights, V, "... queries keys,  ... keys d_v -> ... queries d_v")
    return output


class MultiheadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        super().__init__()

        self.d_k = d_model
        self.d_v = d_model
        self.num_heads = num_heads
        self.dim_per_head = self.d_k // num_heads

        self.q_proj = LinearNobias(self.d_k, d_model, device, dtype)
        self.k_proj = LinearNobias(self.d_k, d_model, device, dtype)
        self.v_proj = LinearNobias(self.d_v, d_model, device, dtype)

        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.dim_per_head, max_seq_len, device)
        else:
            self.rope = None

        self.out_proj = LinearNobias(d_model, self.d_v, device, dtype)

    @staticmethod
    def create_causal_mask(seq_len: int, batch_dims: tuple) -> Tensor:
        '''
        Lower Triangular Matrix Creation : 
            Create a boolean matrix of size (seq_len, seq_len) 
            where each element (i, j) is True 
            if j <= i (indicating that position i can attend to position j).
        Reshaping for Batch Dimensions : 
            Reshape the matrix to include leading singleton dimensions corresponding to the batch dimensions, 
            allowing for proper broadcasting.
        Expanding to Batch Size : 
            Expand the reshaped matrix to match the provided batch dimensions, 
            ensuring the causal mask is applied uniformly across all batch elements.
        '''
        # Create a lower triangular boolean matrix of shape (seq_len, seq_len)
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0)

        # Generate row and column indices using broadcasting
        # i = torch.arange(seq_len).view(seq_len, 1)
        # j = torch.arange(seq_len).view(1, seq_len)
        # mask = (i >= j)  # Shape (seq_len, seq_len)

        # Reshape to add leading singleton dimensions for batch_dims
        mask = mask.view((1,) * len(batch_dims) + (seq_len, seq_len))
        # Expand the mask to match the batch dimensions
        mask = mask.expand(batch_dims + (seq_len, seq_len))
        return mask

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_in"],
    ) -> Float[Tensor, " ... sequence_length d_out"]:

        seq_len = x.size(-2)
        batch_dims = x.shape[:-2]

        q = einops.rearrange(self.q_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        k = einops.rearrange(self.k_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)
        v = einops.rearrange(self.v_proj(x), '... s (h d) -> ... h s d', h=self.num_heads)

        # 动态生成位置编码
        if self.rope is not None:
            # 生成绝对位置索引 (形状: batch_dims + (1, seq_len))
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.view((1,)*len(batch_dims) + (seq_len,))
            token_positions = token_positions.expand(batch_dims + (seq_len,)).unsqueeze(-2)

            # 应用旋转编码到每个位置
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = self.create_causal_mask(seq_len, batch_dims)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        return self.out_proj(einops.rearrange(attn_out, '... h s d -> ... s (h d)'))
