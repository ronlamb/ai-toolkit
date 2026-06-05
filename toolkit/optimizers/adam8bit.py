import math
import torch
from torch.optim import Optimizer
from toolkit.optimizers.optimizer_utils import copy_stochastic, Auto8bitTensor, stochastic_grad_accummulation

class Adam8bit(Optimizer):
    """
    Implements Adam optimizer with 8-bit state storage and stochastic rounding.
    
    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        lr (float): Learning rate (default: 1e-3)
        betas (tuple): Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float): Term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay (float): Weight decay coefficient (default: 0)
        decouple (bool): Use AdamW style decoupled weight decay (default: True)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, decouple=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       decouple=decouple)
        super(Adam8bit, self).__init__(params, defaults)
        
        self.is_stochastic_rounding_accumulation = False
        
        # Setup stochastic grad accumulation hooks
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        stochastic_grad_accummulation
                    )

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # Copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        self.step_hook()

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            decay = group['weight_decay']
            decouple = group['decouple']

            # group-level step
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1
            step = group['step']

            # fused bias correction
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            inv_bc1 = 1.0 / bc1
            inv_bc2_sqrt = 1.0 / math.sqrt(bc2)
            step_size = lr * inv_bc1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['fp32_buffer'] = torch.zeros_like(p, dtype=torch.float32)

                    # persistent fp32 EMA buffers
                    state['exp_avg_fp32'] = torch.zeros_like(p, dtype=torch.float32)
                    state['exp_avg_sq_fp32'] = torch.zeros_like(p, dtype=torch.float32)

                    state['exp_avg'] = Auto8bitTensor(
                        torch.zeros_like(state['fp32_buffer'])
                    )
                    state['exp_avg_sq'] = Auto8bitTensor(
                        torch.zeros_like(state['fp32_buffer'])
                    )

                p_fp32 = state['fp32_buffer']
                p_fp32.copy_(p, non_blocking=True)

                # reuse fp32 EMA buffers (no new allocations)
                exp_avg = state['exp_avg_fp32']
                exp_avg.copy_(state['exp_avg'].dequantize(), non_blocking=True)

                exp_avg_sq = state['exp_avg_sq_fp32']
                exp_avg_sq.copy_(state['exp_avg_sq'].dequantize(), non_blocking=True)

                state['step'] = step

                # fused EMA + quantization (first and second moments)
                state['exp_avg'].update_from_fp32_(
                    exp_avg, fused=True, beta=beta1, grad=grad
                )
                state['exp_avg_sq'].update_from_fp32_(
                    exp_avg_sq, fused=True, beta=beta2, grad=grad  # grad² fused inside
                )

                # decoupled weight decay
                if decay != 0 and decouple:
                    p_fp32.mul_(1 - lr * decay)

                # denom in-place: sqrt, bias correction, +eps
                exp_avg_sq.sqrt_().mul_(inv_bc2_sqrt).add_(eps)

                # parameter update (uses exp_avg_sq as denom)
                p_fp32.addcdiv_(exp_avg, exp_avg_sq, value=-step_size)

                # stochastic rounding to parameters
                copy_stochastic(p.data, p_fp32.data)

        return loss
        
        def state_dict(self):
            """Returns the state of the optimizer as a dict."""
            state_dict = super().state_dict()
            
            # Convert Auto8bitTensor objects to regular state dicts
            for param_id, param_state in state_dict['state'].items():
                for key, value in param_state.items():
                    if isinstance(value, Auto8bitTensor):
                        param_state[key] = {
                            '_type': 'Auto8bitTensor',
                            'state': value.state_dict()
                        }
            
            return state_dict

        def load_state_dict(self, state_dict):
            """Loads the optimizer state."""
            # First, load the basic state
            super().load_state_dict(state_dict)
            
            # Then convert any Auto8bitTensor states back to objects
            for param_id, param_state in self.state.items():
                for key, value in param_state.items():
                    if isinstance(value, dict) and value.get('_type') == 'Auto8bitTensor':
                        param_state[key] = Auto8bitTensor(value['state'])

