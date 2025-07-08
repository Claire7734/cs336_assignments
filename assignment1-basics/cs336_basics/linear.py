import torch
import math
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from einops import einsum

def init_trunc_normal_linear(weight: torch.Tensor):
    """
    Initializes the weight with truncated normal:
    N(0, 2 / (din + dout)), truncated at ±3σ
    """
    # weight shape: (dout, din)
    dout, din = weight.shape
    std = math.sqrt(2.0 / (din + dout))
    a, b = -3 * std, 3 * std  # truncation bounds

    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=a, b=b)

class Linear_nobias(nn.Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        init_trunc_normal_linear(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        # input @ self.weight.T
        # input: (..., d_in)
        # weight: (d_out, d_in)
        return einsum(input, self.weight, "... d_in, d_out d_in -> ... d_out")
