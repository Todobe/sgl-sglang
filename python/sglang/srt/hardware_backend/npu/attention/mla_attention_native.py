"""Torch-native fallback for Ascend FIA / MLA attention.

This is a correctness/debugging implementation intended to replace
`torch.ops.npu.npu_fused_infer_attention_score(...)` with pure PyTorch ops.

It supports the two cases you showed:

1) Decode (paged/page-attention), input_layout="BSND":
   - query:      (B, Sq, Nq, Dq)
   - key/value:  KV cache in blocks, e.g. (blockNum, blockSize, Nk*Dk)
   - key_rope:   KV rope cache in blocks, e.g. (blockNum, blockSize, Nk*Dr)
   - block_table:(B, maxBlocks)
   - actual_seq_lengths_kv: per-batch KV lengths

2) Prefill (packed tokens), input_layout="TND":
   - query/key/value: (T, N, D)
   - actual_seq_lengths / actual_seq_lengths_kv: either lengths-per-batch
     (sum==T) or cumulative-ends (last==T)

Returned `attn_output` follows the query layout:
   - BSND -> (B, Sq, Nq, Dv)
   - TND  -> (T, Nq, Dv)

NOTE: This is NOT optimized. Expect it to be much slower than the fused NPU op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

Tensor = torch.Tensor


# -------------------------
# Sequence boundary helpers
# -------------------------

def _to_int_list(x: Optional[Union[Sequence[int], Tensor]]) -> Optional[List[int]]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    return [int(v) for v in x]


def _parse_seq_ends(lengths_or_cumsum: Optional[Union[Sequence[int], Tensor]], total_tokens: int) -> Optional[List[int]]:
    """Return cumulative ends (prefix sums) or None.

    Accepts either:
      - cumulative ends: [s0, s0+s1, ...]  (last == total_tokens)
      - lengths:         [s0, s1, ...]     (sum  == total_tokens)

    This matches how different callers pass `actual_seq_lengths` for TND.
    """
    arr = _to_int_list(lengths_or_cumsum)
    if arr is None or len(arr) == 0:
        return None

    if arr[-1] == total_tokens:
        # already cumulative
        return arr

    # treat as lengths
    ends: List[int] = []
    s = 0
    for l in arr:
        s += int(l)
        ends.append(s)
    if s != total_tokens:
        raise ValueError(f"Sum(actual_seq_lengths)={s} != total_tokens={total_tokens}")
    return ends


def _seq_ranges_from_ends(ends: Optional[List[int]], total_tokens: int) -> List[Tuple[int, int]]:
    if ends is None:
        return [(0, total_tokens)]
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


# -------------------------
# KV layout helpers
# -------------------------

@dataclass
class _KvTokenMajor:
    data: Tensor  # (L, Nk, D)


def _as_token_major_split_heads(
    x: Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
    token_dim_first_for_blocks: bool,
) -> _KvTokenMajor:
    """Convert common KV/cache layouts into (L, Nk, D).

    Supported:
      - (L, Nk, D)
      - (L, Nk*D)
      - (A, B, Nk*D)  (blocks or [B,S,*]) -> flattened into L=A*B
      - (A, Nk, B, D) (blocks) -> flattened into L=A*B
      - (A, B, Nk, D) -> flattened

    `token_dim_first_for_blocks=True` means x is a cache block tensor where the
    first dim is blockNum; we always flatten blocks into tokens.
    """
    if x.dim() == 2:
        # (L, Nk*D)
        if x.shape[-1] != num_kv_heads * head_dim:
            raise ValueError(
                f"KV 2D last dim must be Nk*D ({num_kv_heads}*{head_dim}={num_kv_heads*head_dim}), got {x.shape[-1]}"
            )
        return _KvTokenMajor(x.reshape(-1, num_kv_heads, head_dim))

    if x.dim() == 3:
        # (L, Nk, D)  or  (A, B, Nk*D)
        if x.shape[-1] == head_dim and x.shape[-2] == num_kv_heads:
            return _KvTokenMajor(x)
        if x.shape[-1] == num_kv_heads * head_dim:
            # (A, B, Nk*D) -> flatten A*B
            return _KvTokenMajor(x.reshape(-1, num_kv_heads, head_dim))
        raise ValueError(f"Unsupported KV 3D shape {tuple(x.shape)} for Nk={num_kv_heads}, D={head_dim}")

    if x.dim() == 4:
        # (blockNum, Nk, blockSize, D)
        if x.shape[-1] == head_dim and x.shape[-3] == num_kv_heads:
            # -> (blockNum, blockSize, Nk, D) -> flatten
            perm = x.permute(0, 2, 1, 3).contiguous()
            return _KvTokenMajor(perm.view(-1, num_kv_heads, head_dim))
        # (A, B, Nk, D)
        if x.shape[-1] == head_dim and x.shape[-2] == num_kv_heads:
            return _KvTokenMajor(x.reshape(-1, num_kv_heads, head_dim))
        raise ValueError(f"Unsupported KV 4D shape {tuple(x.shape)} for Nk={num_kv_heads}, D={head_dim}")

    raise ValueError(f"Unsupported KV tensor rank: {x.dim()}")


def _infer_paged_head_dim_per_head(kv: Tensor, num_kv_heads: int) -> int:
    """Infer per-head D from a paged KV tensor.

    Common paged cache layouts:
      - (blockNum, blockSize, Nk*D)  -> D = last / Nk
      - (blockNum, Nk, blockSize, D)-> D = last
    """
    if kv.dim() == 3:
        if kv.shape[-1] % num_kv_heads != 0:
            raise ValueError(f"Paged KV last dim {kv.shape[-1]} must be divisible by Nk={num_kv_heads}")
        return kv.shape[-1] // num_kv_heads
    if kv.dim() == 4:
        return kv.shape[-1]
    raise ValueError(f"Unsupported paged KV rank {kv.dim()}")


def _gather_paged_kv(
    kv: Tensor,
    *,
    block_table_1d: Tensor,
    block_size: int,
    seqlen_kv: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tensor:
    """Gather one sequence's paged KV cache into (L, Nk, D)."""
    if block_size <= 0:
        raise ValueError("block_size must be > 0 for paged attention")

    num_pages = (seqlen_kv + block_size - 1) // block_size
    pages = block_table_1d[:num_pages].to(dtype=torch.long)

    gathered = kv.index_select(0, pages)  # (num_pages, ...)
    tok = _as_token_major_split_heads(
        gathered,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        token_dim_first_for_blocks=True,
    ).data
    return tok[:seqlen_kv]


# -------------------------
# Attention core
# -------------------------

def _maybe_cat_rope(x: Tensor, rope: Optional[Tensor]) -> Tensor:
    if rope is None:
        return x
    if rope.shape[:-1] != x.shape[:-1]:
        raise ValueError(f"rope shape mismatch: x {tuple(x.shape)} vs rope {tuple(rope.shape)}")
    return torch.cat([x, rope], dim=-1)


def _repeat_kv_for_gqa(kv: Tensor, num_q_heads: int, num_kv_heads: int) -> Tensor:
    # kv: (L, Nk, D) -> (L, Nq, D)
    if num_kv_heads == num_q_heads:
        return kv
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_q_heads} must be divisible by num_kv_heads={num_kv_heads}")
    group = num_q_heads // num_kv_heads
    return kv.repeat_interleave(group, dim=1)


def _causal_mask_like(scores: Tensor) -> Tensor:
    """Create a causal mask for (N, Lq, Lkv) scores."""
    _, lq, lkv = scores.shape
    # allow attend to <= i
    m = torch.full((lq, lkv), float("-inf"), device=scores.device, dtype=scores.dtype)
    m = torch.triu(m, diagonal=1)
    return m.unsqueeze(0)  # (1, Lq, Lkv)


def _slice_atten_mask(atten_mask: torch.Tensor, lq: int, lkv: int, b: int = 0) -> torch.Tensor:
    """
    Normalize common mask shapes to (M, lq, lkv) where M is 1 or N(head/broadcast).

    Handles:
      - 2D:  (Lq_mask, Lkv_mask)
      - 3D:  (M, Lq_mask, Lkv_mask)
      - 4D:  (B, 1, Lq_mask, Lkv_mask) or (1,1,Lq_mask,Lkv_mask)

    Special case:
      If mask is square (W,W) but actual (lq,lkv) is larger and lq==lkv,
      interpret W as a sliding-window size and build a banded causal mask.
      This fixes the common "mask cached at window size" situation.
    """
    m = atten_mask

    # ---- normalize to 3D (M, Lq_mask, Lkv_mask) WITHOUT pre-slicing by lq/lkv ----
    if m.dim() == 2:
        m = m.unsqueeze(0)  # (1, Lq_mask, Lkv_mask)
    elif m.dim() == 3:
        # already (M, Lq_mask, Lkv_mask)
        pass
    elif m.dim() == 4:
        bb = 0 if m.shape[0] == 1 else b
        m = m[bb, 0].unsqueeze(0)  # (1, Lq_mask, Lkv_mask)
    else:
        raise ValueError(f"Unsupported atten_mask rank {m.dim()}")

    Lq_mask, Lkv_mask = m.shape[-2], m.shape[-1]

    # ---- exact / larger mask: just slice to needed region ----
    if Lq_mask >= lq and Lkv_mask >= lkv:
        return m[:, :lq, :lkv]

    # ---- template square mask case: build (M,lq,lkv) sliding causal band ----
    # Typical when a cached W×W causal/window mask is reused but seq len grows beyond W.
    if (Lq_mask == Lkv_mask) and (lq == lkv) and (lq > Lq_mask):
        W = Lq_mask  # treat as window size
        # Build additive mask: 0 for allowed, -inf for masked
        # Allowed: k in [q-(W-1), q] (and k<=q)
        q_pos = torch.arange(lq, device=m.device)
        k_pos = torch.arange(lkv, device=m.device)
        future = k_pos[None, :] > q_pos[:, None]
        too_old = k_pos[None, :] < (q_pos[:, None] - (W - 1))
        band_mask = future | too_old  # True means masked
        full = torch.zeros((m.shape[0], lq, lkv), device=m.device, dtype=m.dtype)
        full.masked_fill_(band_mask.unsqueeze(0), float("-inf"))
        return full

    # ---- general fallback: bottom-right align + pad with -inf ----
    # (Conservative: anything outside provided mask is treated as masked.)
    full = torch.full((m.shape[0], lq, lkv), float("-inf"), device=m.device, dtype=m.dtype)

    # how many rows/cols we can actually copy
    copy_lq = min(Lq_mask, lq)
    copy_lkv = min(Lkv_mask, lkv)

    # align to the bottom-right corner
    q_dst0 = lq - copy_lq
    k_dst0 = lkv - copy_lkv
    q_src0 = Lq_mask - copy_lq
    k_src0 = Lkv_mask - copy_lkv

    full[:, q_dst0:q_dst0 + copy_lq, k_dst0:k_dst0 + copy_lkv] = m[:, q_src0:q_src0 + copy_lq, k_src0:k_src0 + copy_lkv]
    return full


def _attention_single_sequence(
    q: Tensor,  # (Lq, Nq, Dq)
    k: Tensor,  # (Lkv, Nk, Dk)
    v: Tensor,  # (Lkv, Nk, Dv)
    *,
    q_rope: Optional[Tensor],
    k_rope: Optional[Tensor],
    num_heads: int,
    num_key_value_heads: int,
    scale: float,
    atten_mask: Optional[Tensor],
    sparse_mode: int,
) -> Tensor:
    """Return (Lq, Nq, Dv)."""
    q_full = _maybe_cat_rope(q, q_rope)  # (Lq, Nq, Dq+Dr)
    k_full = _maybe_cat_rope(k, k_rope)  # (Lkv, Nk, Dk+Dr)

    # expand kv heads to q heads (GQA/MQA)
    k_e = _repeat_kv_for_gqa(k_full, num_heads, num_key_value_heads)  # (Lkv, Nq, D)
    v_e = _repeat_kv_for_gqa(v, num_heads, num_key_value_heads)       # (Lkv, Nq, Dv)

    # compute attention scores: (Nq, Lq, Lkv)
    qf = q_full.to(torch.float32)
    kf = k_e.to(torch.float32)
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * float(scale)

    # apply mask
    if atten_mask is not None:
        scores = scores + _slice_atten_mask(atten_mask, scores.shape[-2], scores.shape[-1]).to(scores.dtype)

    # sparse_mode=3 is causal in Ascend docs; implement causal on top.
    if int(sparse_mode) == 3:
        scores = scores + _causal_mask_like(scores)
    elif int(sparse_mode) != 0:
        raise NotImplementedError(f"sparse_mode={sparse_mode} not implemented in native fallback (support: 0,3)")

    probs = torch.softmax(scores, dim=-1)  # (Nq, Lq, Lkv)

    vf = v_e.to(torch.float32)
    out = torch.einsum("hqk,khd->qhd", probs, vf)  # (Lq, Nq, Dv)
    return out.to(q.dtype)


# -------------------------
# Public entry (drop-in)
# -------------------------

def fused_infer_attention_score_native(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
    num_heads: int,
    num_key_value_heads: int = 0,
    input_layout: str = "BSND",
    atten_mask: Optional[Tensor] = None,
    sparse_mode: int = 0,
    scale: Optional[float] = None,
    antiquant_mode: int = 0,
    antiquant_scale: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    block_size: int = 0,
    actual_seq_lengths: Optional[Union[Sequence[int], Tensor]] = None,
    actual_seq_lengths_kv: Optional[Union[Sequence[int], Tensor]] = None,
    next_tokens: int = 0,
    pre_tokens: int = 2**31 - 1,
    **kwargs,
) -> Tuple[Tensor, None]:
    """Drop-in replacement for torch.ops.npu.npu_fused_infer_attention_score.

    Returns (attn_output, None). Only implements sparse_mode 0 and 3.
    Quantization/antiquant/pse_shift/prefix/etc are ignored.
    """
    if num_key_value_heads in (0, None):
        num_key_value_heads = num_heads

    layout = input_layout.upper()

    # default scale: 1/sqrt(D_total)
    if scale is None:
        d = query.shape[-1]
        dr = query_rope.shape[-1] if query_rope is not None else 0
        scale = 1.0 / math.sqrt(float(d + dr))

    if layout == "BSND":
        if query.dim() != 4:
            raise ValueError(f"BSND expects query (B,S,N,D), got {tuple(query.shape)}")
        bsz, sq, nq, d_q = query.shape
        if nq != num_heads:
            raise ValueError(f"num_heads={num_heads} must match query N={nq} in BSND")

        act_q = _to_int_list(actual_seq_lengths) or [sq] * bsz

        act_kv = _to_int_list(actual_seq_lengths_kv)
        if act_kv is None:
            if block_table is not None:
                raise ValueError("actual_seq_lengths_kv must be provided for paged BSND")
            # non-paged: infer
            if key.dim() == 4:
                act_kv = [key.shape[1]] * bsz
            else:
                act_kv = [key.shape[0]] * bsz

        outs: List[Tensor] = []
        for b in range(bsz):
            q_b = query[b, : act_q[b]]
            q_rope_b = query_rope[b, : act_q[b]] if query_rope is not None else None

            seqlen_kv = int(act_kv[b])

            if block_table is not None:
                bt_b = block_table[b]

                # key/value head dims
                d_k = d_q
                d_v = _infer_paged_head_dim_per_head(value, num_key_value_heads)
                d_r = _infer_paged_head_dim_per_head(key_rope, num_key_value_heads) if key_rope is not None else 0

                k_b = _gather_paged_kv(
                    key,
                    block_table_1d=bt_b,
                    block_size=block_size,
                    seqlen_kv=seqlen_kv,
                    num_kv_heads=num_key_value_heads,
                    head_dim=d_k,
                )
                v_b = _gather_paged_kv(
                    value,
                    block_table_1d=bt_b,
                    block_size=block_size,
                    seqlen_kv=seqlen_kv,
                    num_kv_heads=num_key_value_heads,
                    head_dim=d_v,
                )
                k_rope_b = (
                    _gather_paged_kv(
                        key_rope,
                        block_table_1d=bt_b,
                        block_size=block_size,
                        seqlen_kv=seqlen_kv,
                        num_kv_heads=num_key_value_heads,
                        head_dim=d_r,
                    )
                    if key_rope is not None
                    else None
                )
            else:
                # Non-paged: accept (B,S,N,D) or (S,N,D) or packed-head last dim
                kb = key[b] if key.dim() == 4 else key
                vb = value[b] if value.dim() == 4 else value
                k_b = _as_token_major_split_heads(kb, num_kv_heads=num_key_value_heads, head_dim=d_q, token_dim_first_for_blocks=False).data[:seqlen_kv]

                # infer Dv: if vb is (S,N,Dv) it's last dim; if packed it's last/N
                if vb.dim() == 3 and vb.shape[-2] == num_key_value_heads:
                    dv = vb.shape[-1]
                elif vb.dim() >= 2 and vb.shape[-1] % num_key_value_heads == 0:
                    dv = vb.shape[-1] // num_key_value_heads
                else:
                    dv = d_q
                v_b = _as_token_major_split_heads(vb, num_kv_heads=num_key_value_heads, head_dim=dv, token_dim_first_for_blocks=False).data[:seqlen_kv]

                if key_rope is not None:
                    krb = key_rope[b] if key_rope.dim() == 4 else key_rope
                    if krb.dim() == 3 and krb.shape[-2] == num_key_value_heads:
                        dr = krb.shape[-1]
                    elif krb.shape[-1] % num_key_value_heads == 0:
                        dr = krb.shape[-1] // num_key_value_heads
                    else:
                        dr = krb.shape[-1]
                    k_rope_b = _as_token_major_split_heads(krb, num_kv_heads=num_key_value_heads, head_dim=dr, token_dim_first_for_blocks=False).data[:seqlen_kv]
                else:
                    k_rope_b = None

            out_b = _attention_single_sequence(
                q_b,
                k_b,
                v_b,
                q_rope=q_rope_b,
                k_rope=k_rope_b,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                scale=float(scale),
                atten_mask=atten_mask,
                sparse_mode=int(sparse_mode),
            )

            # pad back to Sq
            if out_b.shape[0] < sq:
                pad = torch.zeros((sq - out_b.shape[0], num_heads, out_b.shape[-1]), device=out_b.device, dtype=out_b.dtype)
                out_b = torch.cat([out_b, pad], dim=0)
            outs.append(out_b)

        return torch.stack(outs, dim=0), None

    if layout == "TND":
        if query.dim() != 3:
            raise ValueError(f"TND expects query (T,N,D), got {tuple(query.shape)}")
        t, nq, d_q = query.shape
        if nq != num_heads:
            raise ValueError(f"num_heads={num_heads} must match query N={nq} in TND")

        ends_q = _parse_seq_ends(actual_seq_lengths, t)
        ranges_q = _seq_ranges_from_ends(ends_q, t)

        tk = key.shape[0]
        ends_kv = _parse_seq_ends(actual_seq_lengths_kv, tk) if actual_seq_lengths_kv is not None else ends_q
        ranges_kv = _seq_ranges_from_ends(ends_kv, tk)
        if len(ranges_q) != len(ranges_kv):
            raise ValueError("Batch size mismatch between actual_seq_lengths and actual_seq_lengths_kv")

        # convert K/V once
        k_all = _as_token_major_split_heads(key, num_kv_heads=num_key_value_heads, head_dim=d_q, token_dim_first_for_blocks=False).data

        # infer Dv for TND
        if value.dim() == 3 and value.shape[-2] == num_key_value_heads:
            dv = value.shape[-1]
        elif value.shape[-1] % num_key_value_heads == 0:
            dv = value.shape[-1] // num_key_value_heads
        else:
            dv = d_q
        v_all = _as_token_major_split_heads(value, num_kv_heads=num_key_value_heads, head_dim=dv, token_dim_first_for_blocks=False).data

        if key_rope is not None:
            # infer rope dim
            if key_rope.dim() == 3 and key_rope.shape[-2] == num_key_value_heads:
                dr = key_rope.shape[-1]
            elif key_rope.shape[-1] % num_key_value_heads == 0:
                dr = key_rope.shape[-1] // num_key_value_heads
            else:
                dr = key_rope.shape[-1]
            k_rope_all = _as_token_major_split_heads(key_rope, num_kv_heads=num_key_value_heads, head_dim=dr, token_dim_first_for_blocks=False).data
        else:
            k_rope_all = None

        outs: List[Tensor] = []
        for (qs, qe), (ks, ke) in zip(ranges_q, ranges_kv):
            q_b = query[qs:qe]
            q_rope_b = query_rope[qs:qe] if query_rope is not None else None
            k_b = k_all[ks:ke]
            v_b = v_all[ks:ke]
            k_rope_b = k_rope_all[ks:ke] if k_rope_all is not None else None

            out_b = _attention_single_sequence(
                q_b,
                k_b,
                v_b,
                q_rope=q_rope_b,
                k_rope=k_rope_b,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                scale=float(scale),
                atten_mask=atten_mask,
                sparse_mode=int(sparse_mode),
            )
            outs.append(out_b)

        return torch.cat(outs, dim=0), None

    raise NotImplementedError(f"input_layout={input_layout} not supported by native fallback (support: BSND, TND)")
