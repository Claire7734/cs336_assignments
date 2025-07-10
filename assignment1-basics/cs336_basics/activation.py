import torch
import math
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from einops import einsum


class SwiGLU(nn.Module):

    __constants__ = ["in_features", "out_features"]
    d_model: int
    d_ff: int
    w1: Tensor
    w2: Tensor
    w3: Tensor

    def __init__(
        self,
        d_model: int,
        d_ff: int  | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            d_ff_base = (8 / 3) * d_model
            d_ff = int(round(d_ff_base / 64)) * 64
        self.d_ff = d_ff

        self.w1 = nn.Parameter(torch.empty((self.d_ff, self.d_model), **factory_kwargs))
        self.w3 = nn.Parameter(torch.empty((self.d_ff, self.d_model), **factory_kwargs))
        self.w2 = nn.Parameter(torch.empty((self.d_model, self.d_ff), **factory_kwargs))

        self._init_weights()

    def _init_weights(self):
        # gain = nn.init.calculate_gain('silu') # approx 1.1
        nn.init.xavier_uniform_(self.w1, gain=1.1)
        nn.init.xavier_uniform_(self.w3, gain=1.0)

        nn.init.xavier_uniform_(self.w2)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, seq_len, d_model)
        # SwiGLU(x, W1, W2, W3) = 
        # W2 * (SiLU(W1x) ⊙ W3x) = 
        # W2 * (W1x ⋅ σ(W1x)) ⊙ W3x

        # SiLU(W1x) = W1x ⋅ σ(W1x)
        # torch.sigmoid(x @ self.W1.t())
        z = torch.einsum("b s m, f m -> b s f", x, self.w1)
        h_silu = z * torch.sigmoid(z)

        # h3 = x @ self.W3.t()
        h3 = torch.einsum("b s m, f m -> b s f", x, self.w3)

        # gate = h1 * h3
        gate = h_silu * h3

        # output = gate @ self.W2.t()
        output = torch.einsum("b s f, m f -> b s m", gate, self.w2)
        return output
