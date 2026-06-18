import torch
from einops import rearrange
from torch import Tensor

# Flash-Attention 2 (optional)
try:
    from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
    _HAS_FLASH = True
except (ImportError, ModuleNotFoundError):
    _HAS_FLASH = False


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    # mask should have shape [B, H, L, D]
    if _HAS_FLASH and mask is None and q.is_cuda:
        x = flash_attn_func(
            rearrange(q, "B H L D -> B L H D").contiguous(),
            rearrange(k, "B H L D -> B L H D").contiguous(),
            rearrange(v, "B H L D -> B L H D").contiguous(),
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )
        x = rearrange(x, "B L H D -> B H L D")
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    x = rearrange(x, "B H L D -> B L (H D)")
    return x


# Cache for rope omega tensors: (dim, theta, device) -> omega tensor.
# omega is constant for a given (dim, theta) — no need to recompute every call.
_rope_cache: dict[tuple, Tensor] = {}


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    # Use float32 for MPS compatibility (MPS doesn't support float64).
    # CUDA and CPU will use float32 as well, and it is later converted to float32.
    cache_key = (dim, theta, pos.device)
    omega = _rope_cache.get(cache_key)
    if omega is None:
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
        omega = 1.0 / (theta**scale)
        _rope_cache[cache_key] = omega
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    freqs = freqs_cis.to(xq.dtype)
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_out = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape), xk_out.reshape(*xk.shape)
