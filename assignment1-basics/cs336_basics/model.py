import torch
import math
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.parameter import Parameter
import einops

from cs336_basics.embedding import Embedding
from cs336_basics.linear import LinearNobias
from cs336_basics.attention import MultiheadAttention
from cs336_basics.normalization import RMSNorm
from cs336_basics.activation import SwiGLU

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )

        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = LinearNobias(d_model, vocab_size, device, dtype)

    def forward(
        self,
        x,
    ):
        _, sequence_length = x.size()

        # (batch size, sequence_length, d_model)
        x = self.token_embeddings(x)

        for layer in self.layers:
            # (batch size, sequence_length, d_model)
            x = layer(x)

        # (batch size, sequence_length, d_model)
        x = self.ln_final(x)

        # (batch size, sequence_length, vocab_size)
        return self.lm_head(x)
