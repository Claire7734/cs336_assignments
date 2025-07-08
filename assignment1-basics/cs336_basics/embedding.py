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
