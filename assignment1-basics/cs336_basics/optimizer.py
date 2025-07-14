from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


def cross_entropy(logits, targets):
    # Subtract the maximum for numerical stability
    max_logits = logits.max(dim=-1, keepdim=True).values
    logits_stable = logits - max_logits

    # Compute log sum exp over the vocabulary dimension
    log_sum_exp = torch.logsumexp(logits_stable, dim=-1)

    # Gather the target logits
    # Add a trailing dimension to targets to match the gather input
    targets_unsqueezed = targets.unsqueeze(-1)
    target_logits = torch.gather(logits_stable, dim=-1, index=targets_unsqueezed).squeeze(-1)

    # Compute the loss: log_sum_exp - target_logits
    losses = log_sum_exp - target_logits

    # Average over all batch and sequence dimensions
    return losses.mean()


def get_lr_cosine_schedule(t, alpha_max, alpha_min, Tw, Tc):
    if t < Tw:
        return (t / Tw) * alpha_max
    elif t > Tc:
        return alpha_min
    else:
        # Tw <= t <= Tc
        if Tc == Tw:
            return alpha_max
        cos_factor = (t - Tw) / (Tc - Tw) * math.pi
        cos_val = math.cos(cos_factor)
        return alpha_min + 0.5 * (1 + cos_val) * (alpha_max - alpha_min)


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p] # Get state associated with p.
            t = state.get("t", 0) # Get iteration number from the state, or initial value.
            grad = p.grad.data # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
            state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1
                t = state['step']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                denom = v.sqrt().add_(group['eps'])
                p.data.addcdiv_(m, denom, value=-alpha_t)

                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e1)

    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
