import torch
from typing import Tuple, Callable, Optional, Union


@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast similarity computation in chunks to avoid memory issues.
    """
    B, num_src, C = a.shape
    original_dtype = a.dtype

    # Convert to bf16 for computation to improve performance and reduce memory usage
    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    # Process in chunks
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]  # [B, chunk_size, C]
        scores_chunk = torch.bmm(a_chunk, b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


def do_nothing(
    x: torch.Tensor,
    extra_tensors=None,
    extra_tensors_2=None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Identity function for when merging is disabled."""
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    elif extra_tensors is not None:
        return x, extra_tensors
    else:
        return x


def token_merge_pi3(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Token merging function specifically designed for PI3 architecture.
    
    PI3 uses alternating attention patterns:
    - Even blocks: (B*N, hw, C) - frame attention
    - Odd blocks: (B, N*hw, C) - global attention
    
    This function handles the global attention case where we need to merge tokens
    across frames while preserving the first frame as reference.
    
    Args:
        metric [B, N*hw, C]: Tensor for similarity computation
        w: Image width in tokens (patch grid width)
        h: Image height in tokens (patch grid height) 
        sx: dst stride in x dimension
        sy: dst stride in y dimension
        r: Number of tokens to remove through merging
        no_rand: If True, disable randomness
        generator: Random number generator
        enable_protection: If True, protect top 10% of tokens from merging
        
    Returns:
        (merge, unmerge): Two functions for merging and restoring tokens
    """
    B, N_total, _ = metric.shape  # B=batch, N_total=total tokens across all frames
    
    if r <= 0:
        return do_nothing, do_nothing

    gather = torch.gather
    
    # Calculate tokens per frame: register tokens + patch tokens
    tokens_per_frame = w * h + 5  # 5 register tokens + patch tokens
    num_frames = N_total // tokens_per_frame
    
    # Handle edge case where we don't have complete frames
    if tokens_per_frame * num_frames != N_total:
        # If we have a single frame with fewer tokens, adjust
        if N_total < tokens_per_frame:
            # Single frame with reduced tokens
            num_frames = 1
            tokens_per_frame = N_total
        else:
            raise ValueError(f"Token count mismatch: {N_total} != {tokens_per_frame} * {num_frames}. Expected tokens per frame: {tokens_per_frame}, got {N_total} total tokens for {num_frames} frames")

    with torch.no_grad():
        # Determine protected tokens if enabled
        if enable_protection:
            num_protected = int(N_total * 0.1)
            step = max(1, N_total // num_protected)
            protected_indices = torch.arange(0, N_total, step, device=metric.device)[:num_protected]
        else:
            protected_indices = None
            num_protected = 0

        # Initialize token classification buffer
        idx_buffer_seq = torch.zeros(N_total, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx  # Number of blocks within each frame

        # Mark first frame entirely as destination (reference frame)
        if num_frames > 0:
            idx_buffer_seq[:tokens_per_frame] = -1

        # Process other frames - mark register tokens as dst, patch tokens based on sampling
        if num_frames > 1:
            # Mark register tokens (first 5 tokens) of each frame as destination
            for frame_idx in range(1, num_frames):
                start_idx = frame_idx * tokens_per_frame
                register_end = min(start_idx + 5, start_idx + tokens_per_frame)
                idx_buffer_seq[start_idx:register_end] = -1
                
                # Process patch tokens for this frame
                patch_start = register_end
                patch_end = start_idx + tokens_per_frame
                effective_h = min(hsy * sy, h)
                effective_w = min(wsx * sx, w)
                effective_grid_size = effective_h * effective_w
                
                if no_rand:
                    # Fixed pattern: mark every sx*sy tokens as destination
                    base_pattern = torch.zeros(effective_grid_size, device=metric.device, dtype=torch.int64)
                    for i in range(0, effective_grid_size, sx * sy):
                        if i < effective_grid_size:
                            base_pattern[i] = -1
                    idx_buffer_seq[patch_start:patch_start + effective_grid_size] = base_pattern
                else:
                    # Random sampling within each (sx, sy) region
                    all_rand_idx = torch.randint(
                        sx * sy,
                        size=(hsy, wsx),
                        device=metric.device,
                        generator=generator,
                    )
                    
                    # Create pattern for this frame
                    frame_pattern = torch.zeros(effective_h, effective_w, device=metric.device, dtype=torch.int64)
                    for i in range(hsy):
                        for j in range(wsx):
                            # Get the random index within this region
                            rand_idx = all_rand_idx[i, j]
                            # Calculate the position in the flattened grid
                            region_start = i * sy * w + j * sx
                            # Mark the selected token as destination
                            if region_start + rand_idx < effective_grid_size:
                                flat_idx = region_start + rand_idx
                                row = flat_idx // w
                                col = flat_idx % w
                                if row < effective_h and col < effective_w:
                                    frame_pattern[row, col] = -1
                    
                    idx_buffer_seq[patch_start:patch_start + effective_grid_size] = frame_pattern.flatten()
        else:
            # Single frame case - apply simple sampling
            if tokens_per_frame > 5:  # Only if we have patch tokens
                # Mark some patch tokens as destination using simple sampling
                patch_tokens = tokens_per_frame - 5
                if patch_tokens > 0:
                    # Simple stride-based sampling for single frame
                    stride = max(1, patch_tokens // (r // 2))  # Roughly sample r//2 tokens
                    for i in range(5, tokens_per_frame, stride):
                        if i < tokens_per_frame:
                            idx_buffer_seq[i] = -1

        # Sort indices to separate src and dst
        rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
        num_dst_orig = int((idx_buffer_seq == -1).sum())

        # Original src and dst indices
        a_idx_orig = rand_idx[:, num_dst_orig:, :]  # src indices
        b_idx_orig = rand_idx[:, :num_dst_orig, :]  # dst indices
        a_idx = a_idx_orig
        b_idx = b_idx_orig

        if enable_protection:
            protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
            num_protected_actual = protected_idx.shape[1]
        else:
            protected_idx = None
            num_protected_actual = 0

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]

        # Define function to split tokens into src, dst, and protected
        def split(x):
            C = x.shape[-1]
            if enable_protection:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                protected = gather(x, dim=1, index=protected_idx.expand(B, num_protected_actual, C))
                return src, dst, protected
            else:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

        # Compute cosine similarity
        metric = metric / metric.norm(dim=-1, keepdim=True)
        if enable_protection:
            a, b, protected = split(metric)
        else:
            a, b = split(metric)

        r = min(a.shape[1], r)
        num_src_actual = a.shape[1]
        chunk_size = min(5000, num_src_actual)

        node_max = torch.empty(B, num_src_actual, device=a.device, dtype=a.dtype)
        node_idx = torch.empty(B, num_src_actual, device=a.device, dtype=torch.long)

        b_transposed = b.transpose(-1, -2)
        node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # Filter out protected tokens if protection is enabled
        if enable_protection:
            src_indices = a_idx[0, :, 0]
            protected_mask_src = torch.isin(src_indices, protected_indices)
            edge_flat = edge_idx[0, :, 0]
            valid_mask = ~protected_mask_src[edge_flat]
            valid_edges = edge_flat[valid_mask]

            valid_count = valid_edges.shape[0]
            r_actual = min(r, valid_count)

            unm_idx = valid_edges[r_actual:].unsqueeze(0).unsqueeze(-1)
            src_idx = valid_edges[:r_actual].unsqueeze(0).unsqueeze(-1)
        else:
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            r_actual = r

        # Get dst token indices corresponding to each src token to be merged
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        r = r_actual

    # Define merge function
    def merge(
        x: torch.Tensor,
        mode: str = "mean",
        extra_tensors=None,
        extra_tensors_2=None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if enable_protection:
            src, dst, protected = split(x)
        else:
            src, dst = split(x)

        n, t1, c = src.shape

        # Extract unmerged src tokens
        unm_len = unm_idx.shape[1]
        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, c))
        src_len = src_idx.shape[1]
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, c), src, reduce=mode)

        # Process extra tensors if provided
        merged_extra_1 = None
        merged_extra_2 = None
        if extra_tensors is not None:
            E_dim = extra_tensors.shape[-1]
            if enable_protection:
                src_e, dst_e, protected_e = split(extra_tensors)
            else:
                src_e, dst_e = split(extra_tensors)

            src_e_r = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, E_dim))
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, E_dim))

            dst_e = dst_e.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim), src_e_r, reduce=mode
            )
            if enable_protection:
                merged_extra_1 = torch.cat([unm_e, dst_e, protected_e], dim=1)
            else:
                merged_extra_1 = torch.cat([unm_e, dst_e], dim=1)

        if extra_tensors_2 is not None:
            E_dim_2 = extra_tensors_2.shape[-1]
            if enable_protection:
                src_e2, dst_e2, protected_e2 = split(extra_tensors_2)
            else:
                src_e2, dst_e2 = split(extra_tensors_2)

            src_e2_r = gather(src_e2, dim=-2, index=src_idx.expand(n, src_len, E_dim_2))
            unm_e2 = gather(src_e2, dim=-2, index=unm_idx.expand(n, unm_len, E_dim_2))

            dst_e2 = dst_e2.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim_2), src_e2_r, reduce=mode
            )
            if enable_protection:
                merged_extra_2 = torch.cat([unm_e2, dst_e2, protected_e2], dim=1)
            else:
                merged_extra_2 = torch.cat([unm_e2, dst_e2], dim=1)

        if enable_protection:
            main_result = torch.cat([unm, dst, protected], dim=1)
        else:
            main_result = torch.cat([unm, dst], dim=1)

        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main_result, merged_extra_1, merged_extra_2
        elif merged_extra_1 is not None:
            return main_result, merged_extra_1
        else:
            return main_result

    # Define unmerge function
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]

        if enable_protection:
            protected = x[
                ..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :
            ]

        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        out = torch.zeros(B, N_total, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
            ).expand(B, unm_len, c),
            src=unm,
        )

        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
            ).expand(B, src_len, c),
            src=src,
        )

        if enable_protection:
            out.scatter_(
                dim=-2,
                index=protected_idx.expand(B, num_protected_actual, c),
                src=protected,
            )

        return out

    return merge, unmerge
