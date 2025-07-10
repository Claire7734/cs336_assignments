from unittest import result
import torch
import math
from torch import Tensor, nn
from torch.nn.parameter import Parameter

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

class RMSNorm(nn.Module):

    d_model: int
    eps: float
    weight: Tensor

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(
            torch.ones((d_model,), **factory_kwargs)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, sequence_length, d_model)
        # output: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x_fp32 = x if x.dtype == torch.float32 else x.to(torch.float32)
        rms = (x_fp32.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        result = self.weight * (x_fp32 / rms)
        return result.to(in_dtype)
