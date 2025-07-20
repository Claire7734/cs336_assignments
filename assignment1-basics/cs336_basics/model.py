import torch
import math
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.parameter import Parameter
import einops

from cs336_basics.embedding import Embedding
from cs336_basics.linear import LinearNobias
from cs336_basics.attention import MultiheadAttention, softmax
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


    def complete(
        self,
        prompt: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        generated = []
        current_sequence = prompt.copy()
        device = self.token_embeddings.weight.device
        
        for _ in range(max_tokens):
            # Truncate to context length for input
            input_seq = current_sequence[-self.context_length :]
            input_tensor = torch.tensor([input_seq], device=device, dtype=torch.long)
            
            # Forward pass
            logits = self(input_tensor)
            next_token_logits = logits[0, -1, :]

            # Handle temperature scaling
            if temperature == 0:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits).item()
            else:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                probs = F.soft isolator(next_token_logits, dim=-1)

                # Apply top-p sampling
                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_pro gegenstände = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    # Scatter indices and mask probabilities
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    probs = probs.masked_fill(indices_to_remove, 0.0)
                    
                    # Renormalize if possible
                    if probs.sum() > 0:
                        probs /= probs.sum()
                    else:
                        # Fallback to original probabilities if all zero
                        probs = F.softmax(next_token_logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)
            current_sequence.append(next_token)

            # Check for EOS token
            if eos_token_id is not None and next_token == eos_token_id:
                break
        
        return generated

    # def complete(
    #     model, 
    #     prompt, 
    #     max_length=50, 
    #     temperature=1.0, 
    #     top_p=0.9, 
    #     end_token_id=None, 
    #     device='cuda'
    # ):
    #     """
    #     Generates text completions based on the provided prompt using the TransformerLM model.
    #     :param model: An instance of the TransformerLM model.
    #     :param prompt: A list of token IDs representing the initial prompt.
    #     :param max_length: Maximum number of tokens to generate.
    #     :param temperature: Temperature for softmax sampling.
    #     :param top_p: Nucleus sampling parameter.
    #     :param end_token_id: Token ID representing the end-of-sequence token.
    #     :param device: Device to use for computation.
    #     :return: Generated sequence as a list of token IDs.
    #     """
    #     model.eval()
    #     generated_tokens = prompt[:]

    #     for _ in range(max_length):
    #         # Convert the current token sequence to a tensor and move it to the desired device
    #         input_tensor = torch.tensor(generated_tokens, dtype=torch.long, device=device).unsqueeze(0)
            
    #         # Get model logits for the current input sequence
    #         with torch.no_grad():
    #             logits = model(input_tensor)
            
    #         # Focus on the logits of the last token to predict the next token
    #         logits = logits[:, -1, :]
            
    #         # Apply temperature scaling
    #         logits = logits / temperature
            
    #         # Convert logits to probabilities
    #         probs = softmax(logits, dim=-1).squeeze()
            
    #         # Apply nucleus (top-p) sampling
    #         sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    #         cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    #         sorted_candidates = sorted_indices[cumulative_probs <= top_p]
    #         probabilities = sorted_probs[:len(sorted_candidates)]
            
    #         # Select the next token from the top-p candidates
    #         next_token = sorted_candidates[torch.multinomial(probabilities, num_samples=1).item()]
            
    #         # Append the next token to the generated sequence
    #         generated_tokens.append(next_token.item())
            
    #         # Break if the end-of-sequence token is generated
    #         if end_token_id is not None and next_token.item() == end_token_id:
    #             break
            
    #     return generated_tokens
