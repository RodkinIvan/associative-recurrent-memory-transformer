import torch
from torch.nn.functional import relu as r
import os

def dpfp(x, nu=1):
  x = torch.cat([r(x), r(-x)], dim=-1)
  x_rolled = torch.cat([x.roll(shifts=j, dims=-1)
           for j in range(1,nu+1)], dim=-1)
  x_repeat = torch.cat([x] * nu, dim=-1)
  return x_repeat * x_rolled

class DPFP:
    def __init__(self, nu):
        self.nu = nu
    
    def __call__(self, x):
        nu = self.nu
        x = torch.cat([r(x), r(-x)], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1,nu+1)], dim=-1)
        x_repeat = torch.cat([x] * nu, dim=-1)
        return x_repeat * x_rolled
def attn_mask_to_4d(attn_mask, upper, query_len):
    if attn_mask is None:
        return None
    seg_len = attn_mask.size(-1)
    if upper:
        tri = torch.triu(torch.ones(query_len, seg_len, dtype=attn_mask.dtype, device=attn_mask.device))
    else:
        tri = torch.tril(torch.ones(query_len, seg_len, dtype=attn_mask.dtype, device=attn_mask.device))

    mask = torch.einsum('bj,ij->bij', attn_mask, tri)
    mask = mask.unsqueeze(1)
    return mask

def invert_attn_mask(attn_mask, dtype):
        if os.environ.get("NOT_INVERT_ATTN_MASK"):
            return attn_mask
        min_dtype = torch.finfo(dtype).min
        # Use the same dtype as attn_mask to avoid dtype conversion
        one = torch.tensor(1.0, dtype=torch.long, device=attn_mask.device)
        new_mask = (one - attn_mask.long()) * min_dtype
        return new_mask