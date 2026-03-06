# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from builtins import super
import inspect
from json import load
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# from av import time_base
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import (
    FluxTransformer2DLoadersMixin,
    FromOriginalModelMixin,
    PeftAdapterMixin,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    # apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from utils import EXCEPT_DOUBLE, EXCEPT_SINGLE
from pathlib import Path
from typing import Optional
import logging
SAVE_PATH = "Unsafe_50"
DOUBLE_INDEX = 19
BLOCK_SAVE_PATH = "my_flux/dataset/block_error_ratio_safe2"

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
def _load_U_sub(k: int = 5, device: Optional[torch.device] = None, dtype=torch.float32):
    """
    Load and pack U matrices into four tensors:
      - key_D: shape (num_blocks_D, num_heads, d, k)
      - query_D: same
      - key_S: shape (num_blocks_S, num_heads, d, k)
      - query_S: same

    Returns cpu tensors (you can register_buffer them; later .to(device) will move them).
    """
    num_blocks_D = 19
    num_blocks_S = 38
    num_heads = 24
    d = 128  # the head dimension

    # create containers on CPU
    key_D = torch.zeros((num_blocks_D, num_heads, d, k), dtype=dtype)
    query_D = torch.zeros((num_blocks_D, num_heads, d, k), dtype=dtype)
    key_S = torch.zeros((num_blocks_S, num_heads, d, k), dtype=dtype)
    query_S = torch.zeros((num_blocks_S, num_heads, d, k), dtype=dtype)

    # fill D
    for block_idx in range(num_blocks_D):
        for head_idx in range(num_heads):
            if f"{block_idx}_{head_idx}" not in EXCEPT_DOUBLE:
                path_key = f"U_space/Double_key_{block_idx}/head_{head_idx}_U.pt"
                path_query = f"U_space/Double_query_{block_idx}/head_{head_idx}_U.pt"
                U_key = torch.load(path_key, map_location="cpu")  # expect (d, D)
                U_query = torch.load(path_query, map_location="cpu")
                # slice and store
                key_D[block_idx, head_idx, :, :] = U_key[:, :k].to(dtype)
                query_D[block_idx, head_idx, :, :] = U_query[:, :k].to(dtype)
            else:
                # already zeros -> nothing to do
                pass

    # fill S
    for block_idx in range(num_blocks_S):
        for head_idx in range(num_heads):
            if f"{block_idx}_{head_idx}" not in EXCEPT_SINGLE:
                path_key = f"U_space/Single_key_{block_idx}/head_{head_idx}_U.pt"
                path_query = f"U_space/Single_query_{block_idx}/head_{head_idx}_U.pt"
                U_key = torch.load(path_key, map_location="cpu")
                U_query = torch.load(path_query, map_location="cpu")
                key_S[block_idx, head_idx, :, :] = U_key[:, :k].to(dtype)
                query_S[block_idx, head_idx, :, :] = U_query[:, :k].to(dtype)
            else:
                pass

    print("U matrices packed into tensors (CPU).")
    return key_D, query_D, key_S, query_S

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # print("stop")
    # raise ValueError(f"sequence_dim= but should be 1 or 2.")
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(
                -2
            )  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def apply_safe_rotary_emb(
    x: torch.Tensor,
    block_idx: int,
    # head_idx: int,
    type_str: str = "single",
    k: int = 5,
    type_emb: str = "key",
    threshold: float = 0.8,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Apply q' = (I - r * P) q where P = U_sub U_sub^T and r = ||P q||^2 / ||q||^2 per token.
    Supports batch dimension.
    Args:
        x: Tensor shape (B, seq_len, num_heads, head_dim)
        block_idx, head_idx, type_str: used to locate U file
        k: number of columns of U to use
        threshold: only apply for tokens with r > threshold (others keep unchanged)
        eps: numerical stability
    Returns:
        new_x: Tensor with same shape as x
    """
    assert x.ndim == 4, "x should be (B, seq_len, num_heads, head_dim)"
    B, seq_len, num_heads, head_dim = x.shape
    device = x.device
    dtype = x.dtype
    if len(_U_query_cache_S) == 0:
        _load_U_sub(k=k, device=device, dtype=torch.float32)
    
    if type_str == "single":
        if type_emb == "key":
            U_sub = _U_key_cache_S[f"{block_idx}"]  # 24 (d, k)
        else:
            U_sub = _U_query_cache_S[f"{block_idx}"]  # 24 (d, k)
    else:
        if type_emb == "key":
            U_sub = _U_key_cache_D[f"{block_idx}"]
        else:   
            U_sub = _U_query_cache_D[f"{block_idx}"]
    # extract head vectors: shape (B, seq_len, d)
    q = x.to(torch.float32)  # promote to float32 for stable compute

    # q: (B, seq_len, num_heads, head_dim)
    # U_sub: (num_heads, head_dim, k)
    U_sub = U_sub.unsqueeze(0).expand(B, -1, -1, -1)  # (B, num_heads, head_dim, k)
    q = q.permute(0,2,1,3)  # (B, num_heads, seq_len, head_dim)
    q = q.reshape(B * num_heads, seq_len, head_dim)  # (B*num_heads, seq_len, head_dim)
    U_sub = U_sub.reshape(B * num_heads, head_dim, k)  #(B*num_heads, head_dim, k)
    U_sub = U_sub.to(device)
    coeffs = torch.matmul(q, U_sub)  # (B*num_heads, seq_len, k)
    proj = torch.matmul(coeffs, U_sub.transpose(1,2))  # (B*num_heads, seq_len, head_dim)  == P q
    # proj = U_sub @ U_sub.T @ q.T  # (seq_len, d)
    

    proj_norm_sq = (proj ** 2).sum(dim=-1)  # (B*num_heads, seq_len)
    q_norm_sq = (q ** 2).sum(dim=-1).clamp_min(eps)  # (B*num_heads, seq_len)
    r = proj_norm_sq / q_norm_sq  # (B*num_heads, seq_len)

    # clamp r to [0,1]
    r = r.clamp(min=0.0, max=1.0)

    # thresholding: only apply when r > threshold, otherwise keep q unchanged
    mask = r > threshold  # bool (B*num_heads, seq_len)
    r = r * mask.to(r.dtype)  # zero out r where below threshold
    if mask.any():
        # r = r / r.sum(dim=-1, keepdim=True)
        # compute scaled reconstructions: r.unsqueeze(-1) * proj
        scaled_proj = proj * r.unsqueeze(-1)  # (B*num_heads, seq_len, d)
        q_safe = q - scaled_proj
        del scaled_proj
    else:
        q_safe = q  # no token passes threshold
    q_safe = q_safe.reshape(B, num_heads, seq_len, head_dim).permute(0,2,1,3)  # (B, seq_len, num_heads, head_dim)
    # write back into new tensor (preserve original dtype)
    new_x = x.clone()
    if type_str == "single":
        new_x[:, 4096:, :, :] = (q_safe[:, 4096:, :, :]*0.8).to(dtype).to(device)
        new_x[:, :4096, :, :] = q_safe[:, :4096, :, :].to(dtype).to(device)
    else:
        new_x[:, :, :, :] = (q_safe*1.1).to(dtype).to(device)
    del U_sub, q, coeffs, proj, proj_norm_sq, q_norm_sq, r, mask, q_safe

    return new_x

def apply_orthogonal_safe_rotary_emb(
    Q: torch.Tensor,
    K: torch.Tensor,
    block_idx: int,
    # head_idx: int,
    S_param: torch.Tensor,
    type_str: str = "single",
    k_: int = 5,
    threshold: float = 0.8,
    eps: float = 1e-9,
    _U_query_cache_S : torch.Tensor = None,
    _U_key_cache_S : torch.Tensor = None,
    _U_query_cache_D : torch.Tensor = None,
    _U_key_cache_D : torch.Tensor = None,
    time_step : int = 1000,
) -> torch.Tensor:
    """
    Apply q' = (I - r * P) q where P = U_sub U_sub^T and r = ||P q||^2 / ||q||^2 per token.
    Supports batch dimension.
    Args:
        x: Tensor shape (B, seq_len, num_heads, head_dim)
        block_idx, head_idx, type_str: used to locate U file
        k: number of columns of U to use
        threshold: only apply for tokens with r > threshold (others keep unchanged)
        eps: numerical stability
    Returns:
        new_x: Tensor with same shape as x
    """
    assert Q.ndim == 4, "x should be (B, seq_len, num_heads, head_dim)"
    B, seq_len, num_heads, head_dim = Q.shape
    device = Q.device
    dtype = Q.dtype    
    # dtype = torch.float16
    if type_str == "single":
        U_sub_K = _U_key_cache_S[block_idx,:,:,:]  # 24 (d, k)
        U_sub_Q = _U_query_cache_S[block_idx,:,:,:]  # 24 (d, k)
    else:
        U_sub_K = _U_key_cache_D[block_idx,:,:,:]
        U_sub_Q = _U_query_cache_D[block_idx,:,:,:]
    # extract head vectors: shape (B, seq_len, d)
    q = Q.to(U_sub_K.dtype)  # promote to float32 for stable compute
    k = K.to(U_sub_K.dtype)  # promote to float32 for stable compute
    # q: (B, seq_len, num_heads, head_dim)
    # U_sub: (num_heads, head_dim, k)
    U_sub_Q = U_sub_Q.unsqueeze(0).expand(B, -1, -1, -1)  # (B, num_heads, head_dim, k)
    q = q.permute(0,2,1,3)  # (B, num_heads, seq_len, head_dim)
    q = q.reshape(B * num_heads, seq_len, head_dim)  # (B*num_heads, seq_len, head_dim)
    U_sub_Q = U_sub_Q.reshape(B * num_heads, head_dim, k_)  #(B*num_heads, head_dim, k)
    U_sub_Q = U_sub_Q.to(device)
    coeffs_Q = torch.matmul(q, U_sub_Q)  # (B*num_heads, seq_len, k)
    proj_Q = torch.matmul(coeffs_Q, U_sub_Q.transpose(1,2))  # (B*num_heads, seq_len, head_dim)  == P q
    # proj = U_sub @ U_sub.T @ q.T  # (seq_len, d)

    proj_norm_sq = (proj_Q ** 2).sum(dim=-1)  # (B*num_heads, seq_len)
    q_norm_sq = (q ** 2).sum(dim=-1).clamp_min(eps)  # (B*num_heads, seq_len)
    r_q = proj_norm_sq / q_norm_sq  # (B*num_heads, seq_len)   

    # clamp r to [0,1]
    r_q = r_q.clamp(min=0.0, max=1.0)

    # thresholding: only apply when r > threshold, otherwise keep q unchanged
    mask = r_q > threshold  # bool (B*num_heads, seq_len)

        
    r_q = r_q * mask.to(r_q.dtype)  # zero out r where below threshold
    if mask.any():
        # print("shape of r_q:", r_q.shape)
        # print("shape of S_param:", S_param.shape)
        # r = r / r.sum(dim=-1, keepdim=True)
        S_param = S_param.to(r_q.device)
        if type_str == "double":
            S = r_q[...,None,None] * S_param.repeat(B, 1, 1, 1).view(B * num_heads, k_, k_)[:, None, :, :]  # (B*num_heads, seq_len, k, k)
        else:   
            S = r_q[:,:4096,None,None] * (S_param[0].repeat(B, 1, 1, 1).view(B * num_heads, k_, k_)[:, None, :, :])  # (B*num_heads, seq_len, k, k)
            S_next = r_q[:,4096:,None,None] * S_param[1].repeat(B, 1, 1, 1).view(B * num_heads, k_, k_)[:, None, :, :]  # (B*num_heads, seq_len, k, k)
            S = torch.cat([S, S_next], dim=1)  # (B*num_heads, seq_len, k, k)
            del S_next
        exp_S = torch.matrix_exp(S)  # (B*num_heads, seq_len, k, k)
        # compute scaled reconstructions: U*exp(r*S_param)*U^T+I-U*U^T S_param:[5*5]
        scaled_proj = torch.matmul(coeffs_Q.unsqueeze(-2), exp_S).squeeze(-2)  # (B*num_heads, seq_len, k)
        scaled_proj = torch.matmul(scaled_proj, U_sub_Q.transpose(1,2))  # (B*num_heads, seq_len, head_dim) 
        q_safe = q + scaled_proj - proj_Q  # (B*num_heads, seq_len, head_dim)
        del scaled_proj
    else:
        q_safe = q  # no token passes threshold
    q_safe = q_safe.reshape(B, num_heads, seq_len, head_dim).permute(0,2,1,3)  # (B, seq_len, num_heads, head_dim)

    U_sub_K = U_sub_K.unsqueeze(0).expand(B, -1, -1, -1)  # (B, num_heads, head_dim, k)
    k = k.permute(0,2,1,3)  # (B, num_heads, seq_len, head_dim)
    k = k.reshape(B * num_heads, seq_len, head_dim)  # (B*num_heads, seq_len, head_dim)
    U_sub_K = U_sub_K.reshape(B * num_heads, head_dim, k_)  #(B*num_heads, head_dim, k)
    U_sub_K = U_sub_K.to(device)
    coeffs_K = torch.matmul(k, U_sub_K)  # (B*num_heads, seq_len, k)
    proj_K = torch.matmul(coeffs_K, U_sub_K.transpose(1,2))  # (B*num_heads, seq_len, head_dim)  == P k
    
    proj_norm_sq = (proj_K ** 2).sum(dim=-1)  # (B*num_heads, seq_len)
    k_norm_sq = (k ** 2).sum(dim=-1).clamp_min(eps)  # (B*num_heads, seq_len)
    r_k = proj_norm_sq / k_norm_sq  # (B*num_heads, seq_len)

    r_k = r_k.clamp(min=0.0, max=1.0)
    # thresholding: only apply when r > threshold, otherwise keep k unchanged
    mask = r_k > threshold  # bool (B*num_heads, seq_len)
    r_k = r_k * mask.to(r_k.dtype)  # zero out r where below threshold

    if mask.any():  
        # r = r / r.sum(dim=-1, keepdim=True)
        # compute scaled reconstructions: U*exp(r*S_param)*U^T+I-U*U^T S_param:[5*5]
        S_param = S_param.to(r_k.device)       
        if type_str == "double":
            S = r_k[...,None,None] * S_param.repeat(B, 1, 1, 1).view(B * num_heads, k_, k_)[:, None, :, :]  # (B*num_heads, seq_len, k, k)
        else:   
            S = r_k[:,:4096,None,None] * (S_param[0].repeat(B, 1, 1, 1).view(B * num_heads, k_, k_)[:, None, :, :])  # (B*num_heads, seq_len, k, k)
            S_next = r_k[:,4096:,None,None] * S_param[1].repeat(B, 1, 1, 1).view(B * num_heads, k_, k_)[:, None, :, :]  # (B*num_heads, seq_len, k, k)
            S = torch.cat([S, S_next], dim=1)  # (B*num_heads, seq_len, k, k)
            del S_next
        # S = r_k[...,None,None] * S_param.repeat(B, 1, 1, 1).view(B * num_heads, k_, k_)[:, None, :, :]  # (B*num_heads, seq_len, k, k)
        exp_S = torch.matrix_exp(S)  # (B*num_heads, seq_len, k, k)
        scaled_proj = torch.matmul(coeffs_K.unsqueeze(-2), exp_S).squeeze(-2)  # (B*num_heads, seq_len, k)
        scaled_proj = torch.matmul(scaled_proj, U_sub_K.transpose(1,2))  # (B*num_heads, seq_len, head_dim)
        k_safe = k + scaled_proj - proj_K  # (B*num_heads, seq_len, head_dim)
        del scaled_proj, S, exp_S
        
    else:
        k_safe = k  # no token passes threshold
    k_safe = k_safe.reshape(B, num_heads, seq_len, head_dim).permute(0,2,1,3)  # (B, seq_len, num_heads, head_dim)
    S_param = S_param.to("cpu")

    del q, k, proj_Q, proj_K, coeffs_Q, coeffs_K, r_q, r_k, mask, U_sub_Q, U_sub_K
    return q_safe.to(dtype).to(device), k_safe.to(dtype).to(device)


def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(
    attn: "FluxAttention", hidden_states, encoder_hidden_states=None
):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(
            encoder_hidden_states
        ).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(
    attn: "FluxAttention", hidden_states, encoder_hidden_states=None
):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)

class RotaryAdapter(nn.Module):
    def __init__(self, config, r=5, dtype=torch.float16):
        super().__init__()
        self.r = r
        self.num_layers = config.num_layers
        self.num_single_layers = getattr(config, "num_single_layers", 0)
        self.num_heads = config.num_attention_heads

        self.S_param_Double = nn.ParameterList()
        for i in range(self.num_layers):
            # Construct a head*r*r antisymmetric trainable matrix for each layer
            S_init = torch.randn(self.num_heads, self.r, self.r)
            S_init =  (S_init - S_init.transpose(-1, -2))
            S_init = S_init.to(dtype)
            self.S_param_Double.append(nn.Parameter(S_init))

        self.S_param_Single = nn.ParameterList()
        for i in range(self.num_single_layers):
            # Construct a head*r*r antisymmetric trainable matrix for each single layer
            S_init = torch.randn(2, self.num_heads, self.r, self.r)
            S_init[0] *= 0.01  # Initialize the first half with smaller values
            S_init =  (S_init - S_init.transpose(-2, -1))
            
            S_init = S_init.to(dtype) 
            self.S_param_Single.append(nn.Parameter(S_init))
        _U_key_cache_D,_U_query_cache_D,_U_key_cache_S,_U_query_cache_S = _load_U_sub(k=self.r, device=torch.device("cpu"), dtype=dtype)
        # Store the loaded U matrices in buffers to avoid loading them every time forward is called
        self.register_buffer("_U_key_cache_D", _U_key_cache_D)
        self.register_buffer("_U_query_cache_D", _U_query_cache_D)
        self.register_buffer("_U_key_cache_S", _U_key_cache_S)
        self.register_buffer("_U_query_cache_S", _U_query_cache_S)

    def forward(self, query, key, block_type='double', block_idx=0,k_=2, threshold=0.7, eps=1e-9, time_step=1000):
        if block_type == 'double':
            S_param = self.S_param_Double[block_idx]
        elif block_type == 'single':
            S_param = self.S_param_Single[block_idx]
        else:
            raise ValueError("block_type must be 'double' or 'single'")
        # Implement the logic to adjust query and key using S_param here
        new_query, new_key = apply_orthogonal_safe_rotary_emb(
            Q = query, 
            K = key, 
            block_idx=block_idx,
            S_param=S_param,
            type_str=block_type,
            k_=k_,
            threshold=threshold,
            eps=eps,
            _U_query_cache_S = self._U_query_cache_S,
            _U_key_cache_S = self._U_key_cache_S,
            _U_query_cache_D = self._U_query_cache_D,
            _U_key_cache_D = self._U_key_cache_D,
            time_step=time_step
        )
        return new_query, new_key
class FluxAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        index_block: int,
        single_flag: bool = False,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        # U: Optional[torch.Tensor] = None,
        S_param: Optional[torch.Tensor] = None,
        save_flag: bool = False,
        rotary_adapter: Optional[RotaryAdapter] = None,
        timestep: Optional[int] = None,
        save_error_flag: bool = False,
        image_rotary_emb: Optional[torch.Tensor] = None,
        safe_rotary_flag: bool = False,
    ) -> torch.Tensor:
        # print("shape of encoder_hidden_states:", None if encoder_hidden_states is None else encoder_hidden_states.shape)
        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if single_flag and rotary_adapter is not None:
            query, key = rotary_adapter(
                query,
                key,
                block_type='single',
                block_idx=index_block,
                k_=5,
                threshold=0.7,
                eps=1e-9,
                time_step=timestep,
            )

        if timestep==1000 and  single_flag and S_param is not None:
            query, key = apply_orthogonal_safe_rotary_emb(
                query,
                key,
                block_idx=index_block,
                # head_idx=head_idx,
                S_param=S_param,
                type_str="single",
                k_=2,
                threshold=0.7,
                eps=1e-9,
            )
        
        if timestep > 500 and safe_rotary_flag and single_flag:
            # for head_idx in range(query.shape[2]):
            #     if f"{index_block}_{head_idx}" in EXCEPT_SINGLE:
            #         continue
            query = apply_safe_rotary_emb(
                query,
                block_idx=index_block,
                # head_idx=head_idx,
                type_str="single",
                k=5,
                threshold=0.7,
                type_emb="query",
                eps=1e-9,
            )
            # for head_idx in range(key.shape[2]):
            #     if f"{index_block}_{head_idx}" in EXCEPT_SINGLE:
            #         continue
            key = apply_safe_rotary_emb(
                key,
                block_idx=index_block,
                # head_idx=head_idx,
                type_str="single",
                k=5,
                threshold=0.7,
                type_emb="key",
                eps=1e-9,
            )

        if save_flag and single_flag:
            # Save the query and key tensors for the last 512 tokens
            # If the {SAVE_PATH}/Single directory does not exist, create it
            os.makedirs(f"{SAVE_PATH}/Single", exist_ok=True)
            torch.save(
                query.cpu()[0, -512:-462, :, :],    # query: [1, senq_len, heads, head_dim]
                f"{SAVE_PATH}/Single/query_{index_block}.pt",
            )  # Only save the last 512 token vectors
            torch.save(
                key.cpu()[0, -512:-462, :, :],
                f"{SAVE_PATH}/Single/key_{index_block}.pt",
            )
        if save_error_flag and single_flag:
            all_res = []            
            for head_idx in range(query.shape[2]):
                U_space_1 = torch.load(f"U_space/Single_query_{index_block}/head_{head_idx}_U.pt")
                # print("U_space_1 shape:", U_space_1.shape)
                matrix = U_space_1[:, :5].to(torch.float32)  # [head_dim, rank]
                query_head_0 = query[0, 4096:, head_idx, :].clone().detach().to(torch.float32).cpu()  # [senq_len, head_dim]
                norm = matrix @ matrix.T @ query_head_0.T
                mean_error_ratio = (torch.norm(norm) ** 2) / (torch.norm(query_head_0.T) ** 2)
                # Calculate the proportion of tokens with reconstruction error ratio (|UU^Tq|^2/|q|^2) greater than 0.8 among the 512 tokens
                count_half = 0
                for i in range(query_head_0.shape[0]):
                    q_vec = query_head_0[i, :]  # [head_dim]
                    recon_vec = matrix @ matrix.T @ q_vec.T
                    error_ratio = (torch.norm(recon_vec) ** 2) / (torch.norm(q_vec.float()) ** 2)
                    if i == 0:
                        max_error_ratio = error_ratio
                    else:
                        if error_ratio > max_error_ratio:
                            max_error_ratio = error_ratio
                    if error_ratio > 0.8:
                        count_half += 1
                        # print(f"Token {i} Error Ratio: {error_ratio.item()}")
                res = [max_error_ratio.item(), mean_error_ratio.item(), count_half]
                all_res.append(res)
                del U_space_1, matrix, query_head_0, norm, q_vec, recon_vec, error_ratio, max_error_ratio, mean_error_ratio, count_half
            # save the results for all heads of this index_block to a csv file, including index_block, head_idx, max_error_ratio, mean_error_ratio, count_half
            os.makedirs(f"{BLOCK_SAVE_PATH}/Single", exist_ok=True)
            with open(f"{BLOCK_SAVE_PATH}/Single/query_error_ratio_block.csv", "a") as f:
                if f.tell() == 0:
                    f.write("time_step,block_idx,head_idx,max_error_ratio,mean_error_ratio,count_half\n")
                for head_idx in range(query.shape[2]):
                    f.write(f"{timestep.item()},{index_block},{head_idx},{all_res[head_idx][0]},{all_res[head_idx][1]},{all_res[head_idx][2]}\n")
            ################################################################################

            all_res = []
            for head_idx in range(key.shape[2]):
                U_space_1 = torch.load(f"/U_space/Single_key_{index_block}/head_{head_idx}_U.pt")
                # print("U_space_1 shape:", U_space_1.shape)
                matrix = U_space_1[:, :5].to(torch.float32)  # [head_dim, rank]
                key_head_0 = key[0, 4096:, head_idx, :].clone().detach().to(torch.float32).cpu()  # [senq_len, head_dim]
                norm = matrix @ matrix.T @ key_head_0.T
                mean_error_ratio = (torch.norm(norm) ** 2) / (torch.norm(key_head_0.T) ** 2)
                count_half = 0
                for i in range(key_head_0.shape[0]):
                    k_vec = key_head_0[i, :]  # [head_dim]
                    recon_vec = matrix @ matrix.T @ k_vec.T
                    error_ratio = (torch.norm(recon_vec) ** 2) / (torch.norm(k_vec.float()) ** 2)
                    if i == 0:
                        max_error_ratio = error_ratio
                    else:
                        if error_ratio > max_error_ratio:
                            max_error_ratio = error_ratio
                    if error_ratio > 0.8:
                        count_half += 1
                        # print(f"Token {i} Error Ratio: {error_ratio.item()}")
                res = [max_error_ratio.item(), mean_error_ratio.item(), count_half]
                all_res.append(res)
                del U_space_1, matrix, key_head_0, norm, k_vec, recon_vec, error_ratio, max_error_ratio, mean_error_ratio, count_half

            with open(f"{BLOCK_SAVE_PATH}/Single/key_error_ratio_block.csv", "a") as f:
                if f.tell() == 0:   
                    f.write("time_step,block_idx,head_idx,max_error_ratio,mean_error_ratio,count_half\n")
                for head_idx in range(key.shape[2]):
                    f.write(f"{timestep.item()},{index_block},{head_idx},{all_res[head_idx][0]},{all_res[head_idx][1]},{all_res[head_idx][2]}\n")
            del all_res
                ################################################################################

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            if rotary_adapter is not None:
                encoder_query, encoder_key = rotary_adapter(
                    encoder_query,
                    encoder_key,
                    block_type='double',
                    block_idx=index_block,
                    k_=5,
                    threshold=0.7,
                    eps=1e-9,
                    time_step=timestep,
                )
            
            if safe_rotary_flag:
                # for head_idx in range(encoder_query.shape[2]):
                #     if f"{index_block}_{head_idx}" in EXCEPT_DOUBLE:
                #         continue
                encoder_query = apply_safe_rotary_emb(
                    encoder_query,
                    block_idx=index_block,
                    # head_idx=head_idx,
                    type_str="double",
                    k=5,
                    threshold=0.7,
                    type_emb="query",
                    eps=1e-9,
                )
                # for head_idx in range(encoder_key.shape[2]):
                #     if f"{index_block}_{head_idx}" in EXCEPT_DOUBLE:
                #         continue
                encoder_key = apply_safe_rotary_emb(
                    encoder_key,
                    block_idx=index_block,
                    # head_idx=head_idx,
                    type_str="double",
                    k=5,
                    threshold=0.7,
                    type_emb="key",
                    eps=1e-9,
                )
            
            ################################################################################
            if save_error_flag:
                ################################################################################
                # Calculate and save the reconstruction error ratios for encoder_query and encoder_key
                ################################################################################
                all_res = []            
                for head_idx in range(encoder_query.shape[2]):
                    U_space_1 = torch.load(f"U_space/Double_query_{index_block}/head_{head_idx}_U.pt")
                    # print("U_space_1 shape:", U_space_1.shape)
                    matrix = U_space_1[:, :5].to(torch.float32)  # [head_dim, rank]
                    encoder_query_head_0 = encoder_query[0, :, head_idx, :].clone().detach().to(torch.float32).cpu()  # [senq_len, head_dim]
                    norm = matrix @ matrix.T @ encoder_query_head_0.T
                    mean_error_ratio = (torch.norm(norm) ** 2) / (torch.norm(encoder_query_head_0.T) ** 2)
                    count_half = 0
                    for i in range(encoder_query_head_0.shape[0]):
                        q_vec = encoder_query_head_0[i, :]  # [head_dim]
                        recon_vec = matrix @ matrix.T @ q_vec.T
                        error_ratio = (torch.norm(recon_vec) ** 2) / (torch.norm(q_vec.float()) ** 2)
                        if i == 0:
                            max_error_ratio = error_ratio
                        else:
                            if error_ratio > max_error_ratio:
                                max_error_ratio = error_ratio
                        if error_ratio > 0.8:
                            count_half += 1
                            # print(f"Token {i} Error Ratio: {error_ratio.item()}")
                    res = [max_error_ratio.item(), mean_error_ratio.item(), count_half]
                    all_res.append(res)
                    del U_space_1, matrix, encoder_query_head_0, norm, q_vec, recon_vec, error_ratio, max_error_ratio, mean_error_ratio, count_half
                os.makedirs(f"{BLOCK_SAVE_PATH}/Double", exist_ok=True)
                with open(f"{BLOCK_SAVE_PATH}/Double/query_error_ratio_block.csv", "a") as f:
                    if f.tell() == 0:
                        f.write("time_step,block_idx,head_idx,max_error_ratio,mean_error_ratio,count_half\n")
                    for head_idx in range(encoder_query.shape[2]):
                        f.write(f"{timestep.item()},{index_block},{head_idx},{all_res[head_idx][0]},{all_res[head_idx][1]},{all_res[head_idx][2]}\n")
                ################################################################################
                
                all_res = []            
                for head_idx in range(encoder_key.shape[2]):
                    U_space_1 = torch.load(f"U_space/Double_key_{index_block}/head_{head_idx}_U.pt")
                    # print("U_space_1 shape:", U_space_1.shape)
                    matrix = U_space_1[:, :5].to(torch.float32)  # [head_dim, rank]
                    encoder_key_head_0 = encoder_key[0, :, head_idx, :].clone().detach().to(torch.float32).cpu()  # [senq_len, head_dim]
                    norm = matrix @ matrix.T @ encoder_key_head_0.T
                    mean_error_ratio = (torch.norm(norm) ** 2) / (torch.norm(encoder_key_head_0.T) ** 2)
                    count_half = 0
                    for i in range(encoder_key_head_0.shape[0]):
                        k_vec = encoder_key_head_0[i, :]  # [head_dim]
                        recon_vec = matrix @ matrix.T @ k_vec.T
                        error_ratio = (torch.norm(recon_vec) ** 2) / (torch.norm(k_vec.float()) ** 2)
                        if i == 0:
                            max_error_ratio = error_ratio
                        else:
                            if error_ratio > max_error_ratio:
                                max_error_ratio = error_ratio
                        if error_ratio > 0.8:
                            count_half += 1
                            # print(f"Token {i} Error Ratio: {error_ratio.item()}")
                    res = [max_error_ratio.item(), mean_error_ratio.item(), count_half]
                    all_res.append(res)
                    del U_space_1, matrix, encoder_key_head_0, norm, k_vec, recon_vec, error_ratio, max_error_ratio, mean_error_ratio, count_half

                with open(f"{BLOCK_SAVE_PATH}/Double/key_error_ratio_block.csv", "a") as f:
                    if f.tell() == 0:   
                        f.write("time_step,block_idx,head_idx,max_error_ratio,mean_error_ratio,count_half\n")
                    for head_idx in range(encoder_key.shape[2]):
                        f.write(f"{timestep.item()},{index_block},{head_idx},{all_res[head_idx][0]},{all_res[head_idx][1]},{all_res[head_idx][2]}\n")
                del all_res
                ################################################################################
            # save encoder_query, encoder_key tensors
            if save_flag:
                os.makedirs(f"{SAVE_PATH}/Double", exist_ok=True)
                torch.save(
                    encoder_query.cpu()[0, :50, :, :],
                    f"{SAVE_PATH}/Double/encoder_query_{index_block}.pt",
                )
                torch.save(
                    encoder_key.cpu()[0, :50, :, :],
                    f"{SAVE_PATH}/Double/encoder_key_{index_block}.pt",
                )
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, backend=self._attention_backend
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxIPAdapterAttnProcessor(torch.nn.Module):
    """Flux Attention processor for IP-Adapter."""

    _attention_backend = None

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens=(4,),
        scale=1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError(
                "`scale` should be a list of integers with the same length as `num_tokens`."
            )
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [
                nn.Linear(
                    cross_attention_dim,
                    hidden_size,
                    bias=True,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(len(num_tokens))
            ]
        )
        self.to_v_ip = nn.ModuleList(
            [
                nn.Linear(
                    cross_attention_dim,
                    hidden_size,
                    bias=True,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(len(num_tokens))
            ]
        )

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[List[torch.Tensor]] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        ip_query = query

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # IP-adapter
            ip_attn_output = torch.zeros_like(hidden_states)

            for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
            ):
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, attn.head_dim)
                ip_value = ip_value.view(batch_size, -1, attn.heads, attn.head_dim)

                current_ip_hidden_states = dispatch_attention_fn(
                    ip_query,
                    ip_key,
                    ip_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                )
                current_ip_hidden_states = current_ip_hidden_states.reshape(
                    batch_size, -1, attn.heads * attn.head_dim
                )
                current_ip_hidden_states = current_ip_hidden_states.to(ip_query.dtype)
                ip_attn_output += scale * current_ip_hidden_states

            return hidden_states, encoder_hidden_states, ip_attn_output
        else:
            return hidden_states


class FluxAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = FluxAttnProcessor
    _available_processors = [
        FluxAttnProcessor,
        FluxIPAdapterAttnProcessor,
    ]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = torch.nn.RMSNorm(
            dim_head, eps=eps, elementwise_affine=elementwise_affine
        )
        self.norm_k = torch.nn.RMSNorm(
            dim_head, eps=eps, elementwise_affine=elementwise_affine
        )
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([])
            self.to_out.append(
                torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)
            )
            self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_k_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_v_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        index_block: int,
        single_flag: bool = False,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # U: Optional[torch.Tensor] = None,
        S_param: Optional[torch.Tensor] = None,
        save_flag: bool = False,
        image_rotary_emb: Optional[torch.Tensor] = None,
        save_error_flag: bool = False,
        timestep: Optional[int] = None,
        rotary_adapter: Optional[RotaryAdapter] = None,
        safe_rotary_flag: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k
            for k, _ in kwargs.items()
            if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(
            self,
            hidden_states = hidden_states,
            index_block = index_block,
            single_flag = single_flag,
            encoder_hidden_states = encoder_hidden_states,
            attention_mask = attention_mask,
            # U = U,
            S_param = S_param,
            timestep = timestep,
            save_error_flag = save_error_flag,
            save_flag = save_flag,
            rotary_adapter = rotary_adapter,
            image_rotary_emb = image_rotary_emb,
            safe_rotary_flag = safe_rotary_flag,
            **kwargs,
        )


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if is_torch_npu_available():
            from ..attention_processor import FluxAttnProcessor2_0_NPU

            deprecation_message = (
                "Defaulting to FluxAttnProcessor2_0_NPU for NPU devices will be removed. Attention processors "
                "should be set explicitly using the `set_attn_processor` method."
            )
            deprecate("npu_processor", "0.34.0", deprecation_message)
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = FluxAttnProcessor()

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        index_block: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        S_param: Optional[torch.Tensor] = None,
        save_flag: bool = False,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        save_error_flag: bool = False,
        timestep: Optional[int] = None,
        safe_rotary_flag: bool = False,
        rotary_adapter: Optional[RotaryAdapter] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            index_block=index_block,
            single_flag=True,
            hidden_states=norm_hidden_states,
            save_flag=save_flag,
            timestep=timestep,
            save_error_flag=save_error_flag,
            safe_rotary_flag=safe_rotary_flag,
            S_param=S_param,
            image_rotary_emb=image_rotary_emb,
            rotary_adapter=(rotary_adapter if rotary_adapter is not None else None),
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        residual = residual.to(hidden_states.device)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        del residual, norm_hidden_states, mlp_hidden_states, attn_output, gate
        return encoder_hidden_states, hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        index_block: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        S_param: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        save_flag: bool = False,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        save_error_flag: bool = False,
        safe_rotary_flag: bool = False,
        timestep: Optional[int] = None,
        rotary_adapter: Optional[RotaryAdapter] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )
        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.

        attention_outputs = self.attn(
            index_block=index_block,
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            S_param=S_param,
            save_error_flag=save_error_flag,
            timestep=timestep,
            safe_rotary_flag=safe_rotary_flag,
            rotary_adapter=(rotary_adapter if rotary_adapter is not None else None),
            save_flag=save_flag,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states.to(attn_output.device)
        encoder_hidden_states = encoder_hidden_states.to(attn_output.device)
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        del (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
            attn_output,
            context_attn_output,
            ff_output,
            context_ff_output,
        )
        return encoder_hidden_states, hidden_states


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class FluxTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        orig_img_ids: torch.Tensor = None,
        orig_txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        save_flag: bool = False,
        save_error_flag: bool = False,
        safe_rotary_flag: bool = False,
        rotary_adapter: Optional[RotaryAdapter] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # print("FluxTransformer2DModel forward called")
        # apply_rotary_emb(torch.zeros(1,1,1,1), torch.zeros(1,1), sequence_dim=1)
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        # print("check img_ids:", img_ids[10])
        image_rotary_emb = self.pos_embed(ids)
        
        orig_ids = torch.cat((orig_txt_ids, orig_img_ids), dim=0)
        orig_image_rotary_emb = self.pos_embed(orig_ids)
        if (
            joint_attention_kwargs is not None
            and "ip_adapter_image_embeds" in joint_attention_kwargs
        ):
            ip_adapter_image_embeds = joint_attention_kwargs.pop(
                "ip_adapter_image_embeds"
            )
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
        double_index = [0, 2, 1, 4, 6, 3, 5, 8, 7, 10, 9]
        single_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        double_index = []
        single_index = []
        # S_param_D = torch.randn(
        #     len(self.transformer_blocks),
        #     self.config.num_attention_heads,
        #     5,
        #     5,
        #     device=hidden_states.device,
        # )
        # S_param_D = 0.9 * (S_param_D - S_param_D.transpose(-2, -1))
        # S_param_S = torch.randn(
        #     len(self.single_transformer_blocks),
        #     self.config.num_attention_heads,
        #     5,
        #     5,  
        #     device=hidden_states.device,
        # )
        # S_param_S = 0.9 * (S_param_S - S_param_S.transpose(-2, -1))
        
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        orig_image_rotary_emb,
                        joint_attention_kwargs,
                    )
                )

            else:
                if index_block not in double_index:
                    encoder_hidden_states, hidden_states = block(
                        index_block=index_block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        timestep=timestep,
                        safe_rotary_flag=safe_rotary_flag,
                        save_error_flag=save_error_flag,
                        # S_param=S_param_D[index_block],
                        rotary_adapter=(rotary_adapter if rotary_adapter is not None else None),
                        image_rotary_emb=image_rotary_emb,
                        save_flag=save_flag,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                else:
                    # print("shape of encoder_hidden_states:", encoder_hidden_states.shape)
                    encoder_hidden_states, hidden_states = block(
                        index_block=index_block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        timestep=timestep,
                        safe_rotary_flag=safe_rotary_flag,
                        save_error_flag=save_error_flag,
                        # S_param=S_param_D[index_block],
                        rotary_adapter=(rotary_adapter if rotary_adapter is not None else None),
                        image_rotary_emb=orig_image_rotary_emb,
                        save_flag=save_flag,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(
                    controlnet_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[
                            index_block % len(controlnet_block_samples)
                        ]
                    )
                else:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block // interval_control]
                    )

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        orig_image_rotary_emb,
                        joint_attention_kwargs,
                    )
                )

            else:
      
                if index_block not in single_index:
                    encoder_hidden_states, hidden_states = block(
                        index_block=index_block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        timestep=timestep,
                        rotary_adapter=(rotary_adapter if rotary_adapter is not None else None),
                        safe_rotary_flag=safe_rotary_flag,
                        save_error_flag=save_error_flag,
                        # S_param=S_param_S[index_block],
                        save_flag=save_flag,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        index_block=index_block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        timestep=timestep,
                        # S_param=S_param_S[index_block],
                        rotary_adapter=(rotary_adapter if rotary_adapter is not None else None),
                        safe_rotary_flag=safe_rotary_flag,
                        save_error_flag=save_error_flag,
                        save_flag=save_flag,
                        image_rotary_emb=orig_image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states = (
                    hidden_states
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        # output = output.to(device="cuda:4")

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
from collections.abc import Mapping



        