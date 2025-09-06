# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
import time

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend

# Optional Transformer Engine for FP8
try:
    import transformer_engine.pytorch as te  # type: ignore
    from transformer_engine.common import recipe as te_recipe  # type: ignore
    TE_AVAILABLE = True
except Exception:
    TE_AVAILABLE = False

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


    
class FlashAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]

        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v)

        # Use the actual shape after merging/unmerging
        x = x.transpose(1, 2).reshape(B, -1, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


"""
Following is written by GPT-4o
"""
class CrossAttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Separate projection layers for query, key, and value
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, M)
        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(query, key, value, attn_bias)

        B, N, C = query.shape
        _, M, _ = key.shape

        # Project query, key, and value
        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Compute memory-efficient attention
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, C)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.rope = rope

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttentionRope(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        qkv = qkv.transpose(1, 3)
        # q, k, v = unbind(qkv, 2)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(frame_num, 261, frame_num, 261).mean(dim=[1, 3]).sum(1)         # for frame attention matrix
        # global_valid_id = torch.where(score_matrix > 0)
        # score_matrix = (q.permute(0, 2, 1, 3) * self.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class FlashAttentionRope(AttentionRope):
    def forward(self, x: Tensor, attn_bias=None, xpos=None, global_merging=None, merge_funcs=None) -> Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = unbind(qkv, 2)
        # q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        # Apply token merging if enabled for global attention
        merge_num = list(range(36))  # PI3 has 36 decoder blocks
        merging_enabled = merge_funcs is not None
        m_a, u_a = merge_funcs if merge_funcs is not None else (None, None)

        # Check if this is a global attention block (odd-numbered blocks)
        is_global_attention = global_merging is not None and global_merging % 2 == 1

        if global_merging is not None and global_merging in merge_num and is_global_attention:
            # Use pre-calculated merging functions if available
            
            # Apply merging if we have functions
            if m_a is not None and u_a is not None:
                
                # Apply merging to q, k, v
                B_q, H_q, N_q, D_q = q.shape
                q_merge_in = q.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
                k_merge_in = k.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
                v_merge_in = v.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
                
                q_out, k_out, v_out = m_a(
                    q_merge_in,
                    mode="mean",
                    extra_tensors=k_merge_in,
                    extra_tensors_2=v_merge_in,
                )
                
                del q_merge_in, k_merge_in, v_merge_in
                torch.cuda.empty_cache()

                N_m = q_out.shape[1]
                q = q_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
                k = k_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
                v = v_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)

                del q_out, k_out, v_out
                
                torch.cuda.empty_cache()

                merging_enabled = True

        if q.dtype == torch.bfloat16:
            with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v)
        else:
            with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v)

        del q, k, v
        torch.cuda.empty_cache()

        # Use the actual shape after merging/unmerging
        x = x.transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Timing for unmerging
        if global_merging is not None and global_merging in merge_num and u_a is not None:
            x = u_a(x)

        return x


class AttentionRopeFP8(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        rope=None,
        fp8_recipe=None,
    ) -> None:
        super().__init__()
        if not TE_AVAILABLE:
            raise ImportError("Transformer Engine is required for AttentionRopeFP8. Please install 'transformer_engine'.")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Replace Linear with TE Linear for FP8 support
        self.qkv = te.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = te.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.rope = rope

        # Default FP8 recipe
        self.fp8_recipe = fp8_recipe if fp8_recipe is not None else (
            te_recipe.DelayedScaling(margin=0, fp8_format=te_recipe.Format.E4M3) if TE_AVAILABLE else None
        )

    def forward(self, x: Tensor, attn_bias=None, xpos=None, global_merging=None, merge_funcs=None) -> Tensor:
        if not TE_AVAILABLE:
            raise RuntimeError("Transformer Engine is not available for FP8 attention forward path.")

        B, N, C = x.shape

        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)

            q, k, v = unbind(qkv, 2)
            q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

            if self.rope is not None:
                q = self.rope(q, xpos)
                k = self.rope(k, xpos)

            # Apply token merging if enabled for global attention (mirror FlashAttentionRope)
            merge_num = list(range(36))
            merging_enabled = merge_funcs is not None
            m_a, u_a = merge_funcs if merge_funcs is not None else (None, None)
            is_global_attention = global_merging is not None and global_merging % 2 == 1

            if global_merging is not None and global_merging in merge_num and is_global_attention:
                if m_a is not None and u_a is not None:
                    B_q, H_q, N_q, D_q = q.shape
                    q_merge_in = q.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
                    k_merge_in = k.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
                    v_merge_in = v.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)

                    q_out, k_out, v_out = m_a(
                        q_merge_in,
                        mode="mean",
                        extra_tensors=k_merge_in,
                        extra_tensors_2=v_merge_in,
                    )

                    del q_merge_in, k_merge_in, v_merge_in
                    torch.cuda.empty_cache()

                    N_m = q_out.shape[1]
                    q = q_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
                    k = k_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
                    v = v_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)

                    del q_out, k_out, v_out
                    torch.cuda.empty_cache()
                    merging_enabled = True

            # Use SDPA kernels as in FlashAttentionRope
            if q.dtype == torch.bfloat16:
                with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    x = scaled_dot_product_attention(q, k, v)
            else:
                with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                    x = scaled_dot_product_attention(q, k, v)
                
            del q, k, v
            torch.cuda.empty_cache()

            x = x.transpose(1, 2).reshape(B, -1, C)

            # if x is not divisible by 8, pad with zeros, FP8 requires this
            HWN = x.shape[1]
            if HWN % 8 != 0:
                x = F.pad(x, (0, 0, 0, 8 - HWN % 8))
            x = self.proj(x)
            x = self.proj_drop(x)
            
            # unpad 
            x = x[:, :HWN,:]

            if global_merging is not None and global_merging in merge_num and u_a is not None:
                x = u_a(x)

            return x

def get_attn_score(blk_class, x, frame_num, token_length, xpos=None):
    x = blk_class.norm1(x)
    
    B, N, C = x.shape
    qkv = blk_class.attn.qkv(x).reshape(B, N, 3, blk_class.attn.num_heads, C // blk_class.attn.num_heads)
    
    qkv = qkv.transpose(1, 3)
    # q, k, v = unbind(qkv, 2)
    q, k, v = [qkv[:,:,i] for i in range(3)]
    q, k = blk_class.attn.q_norm(q).to(v.dtype), blk_class.attn.k_norm(k).to(v.dtype)

    if blk_class.attn.rope is not None:
        q = blk_class.attn.rope(q, xpos)
        k = blk_class.attn.rope(k, xpos)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    score = (q.permute(0, 2, 1, 3) * blk_class.attn.scale @ k.permute(0, 2, 1, 3).transpose(-2, -1)).sum(dim=1).reshape(B, frame_num, token_length, frame_num, token_length).mean(dim=[2, 4]).sum(-1)

    return score