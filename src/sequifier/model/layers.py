import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sequifier.model.dtypes import cast_floating_to_module_dtype
from sequifier.model.tracing import TraceContext
from sequifier.typechecking import beartype, conditional_beartype


class RMSNorm(nn.Module):
    @beartype
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @conditional_beartype
    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        var = torch.mean(x_fp32.pow(2), dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(var + self.eps)

        return (self.weight.to(x_normed.dtype) * x_normed).to(x.dtype)


class RotaryEmbedding(nn.Module):
    @beartype
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._update_cos_sin_cache(max_seq_len)

    @conditional_beartype
    def _update_cos_sin_cache(self, seq_len):
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        # Different from standard definition to match common implementation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    @conditional_beartype
    def forward(self, x, seq_len):
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[
            :, :, :seq_len, ...
        ]


@conditional_beartype
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@conditional_beartype
def apply_rotary_pos_emb(q, k, cos, sin):
    # Ensure cos/sin match q/k dtype (fix for Mixed Precision/ONNX)
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FeedForward(nn.Module):
    @beartype
    def __init__(self, dim_model, dim_feedforward, activation, dropout):
        super().__init__()
        self.activation = activation

        if activation == "swiglu":
            # SwiGLU requires 2 gates, so we often adjust dim_feedforward or keep it
            # but implement the GLU split. Commonly SwiGLU hidden dim is 2/3 of standard.
            # Here we strictly follow config dim_feedforward.
            self.w1 = nn.Linear(dim_model, dim_feedforward)
            self.w2 = nn.Linear(dim_model, dim_feedforward)  # Gate
            self.w3 = nn.Linear(dim_feedforward, dim_model)  # Output
        else:
            self.linear1 = nn.Linear(dim_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, dim_model)
            self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    @conditional_beartype
    def get_first_layer_dtype(self):
        if self.activation == "swiglu":
            return self.w1.weight.dtype
        else:
            return self.linear1.weight.dtype

    @conditional_beartype
    def forward(self, x, *, trace: TraceContext | None = None, site_prefix: str = ""):
        if self.activation == "swiglu":
            w1_out = self.w1(cast_floating_to_module_dtype(x, self.w1))
            w2_out = self.w2(cast_floating_to_module_dtype(x, self.w2))
            if trace is not None:
                w1_out = trace.emit(
                    f"{site_prefix}.pre_activation",
                    w1_out,
                    axes=("batch", "time", "channel"),
                    width=self.w1.out_features,
                )
            activated = F.silu(w1_out) * w2_out
            if trace is not None:
                activated = trace.emit(
                    f"{site_prefix}.activation",
                    activated,
                    axes=("batch", "time", "channel"),
                    width=self.w1.out_features,
                )
            hidden = self.dropout(activated)
            return self.w3(cast_floating_to_module_dtype(hidden, self.w3))
        else:
            hidden = self.linear1(cast_floating_to_module_dtype(x, self.linear1))
            if trace is not None:
                hidden = trace.emit(
                    f"{site_prefix}.pre_activation",
                    hidden,
                    axes=("batch", "time", "channel"),
                    width=self.linear1.out_features,
                )
            hidden = self.act(hidden)
            if trace is not None:
                hidden = trace.emit(
                    f"{site_prefix}.activation",
                    hidden,
                    axes=("batch", "time", "channel"),
                    width=self.linear1.out_features,
                )
            hidden = self.dropout(hidden)
            return self.linear2(cast_floating_to_module_dtype(hidden, self.linear2))


class SelfAttention(nn.Module):
    @beartype
    def __init__(
        self,
        dim_model,
        n_heads,
        n_kv_heads,
        attention_type,
        dropout,
        context_length,
        use_rope=False,
        rope_theta=10000.0,
        output_projection=True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.dim_model = dim_model
        self.head_dim = dim_model // n_heads
        self.attention_type = attention_type
        self.use_rope = use_rope

        self.wq = nn.Linear(dim_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = (
            nn.Linear(n_heads * self.head_dim, dim_model, bias=False)
            if output_projection
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryEmbedding(
                self.head_dim, max_seq_len=context_length, theta=rope_theta
            )
            if self.head_dim % 2 != 0:
                raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")

    @conditional_beartype
    def forward(
        self,
        x,
        mask=None,
        *,
        trace: TraceContext | None = None,
        site_prefix: str = "",
    ):
        # x shape: (batch, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        xq_input = cast_floating_to_module_dtype(x, self.wq)
        xk_input = cast_floating_to_module_dtype(x, self.wk)
        xv_input = cast_floating_to_module_dtype(x, self.wv)

        xq = (
            self.wq(xq_input)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        xk = (
            self.wk(xk_input)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        xv = (
            self.wv(xv_input)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.use_rope:
            cos, sin = self.rope(xv, seq_len)
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # Handle GQA/MQA by repeating keys/values
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            xk = xk.repeat_interleave(n_rep, dim=1)
            xv = xv.repeat_interleave(n_rep, dim=1)

        if mask is not None and mask.is_floating_point() and mask.dtype != xq.dtype:
            mask = mask.to(dtype=xq.dtype)

        analysis_sites = (
            "q",
            "k",
            "v",
            "scores",
            "weights",
            "update",
        )
        analysis_requested = trace is not None and any(
            trace.requires(f"{site_prefix}.{suffix}") for suffix in analysis_sites
        )
        if not analysis_requested:
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            assert trace is not None
            axes = ("batch", "head", "time", "channel")
            xq = trace.emit(f"{site_prefix}.q", xq, axes=axes, width=self.head_dim)
            xk = trace.emit(f"{site_prefix}.k", xk, axes=axes, width=self.head_dim)
            xv = trace.emit(f"{site_prefix}.v", xv, axes=axes, width=self.head_dim)
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                else:
                    scores = scores + mask
            scores = trace.emit(
                f"{site_prefix}.scores",
                scores,
                axes=("batch", "head", "time", "key_time"),
            )
            weights = torch.softmax(scores, dim=-1)
            weights = trace.emit(
                f"{site_prefix}.weights",
                weights,
                axes=("batch", "head", "time", "key_time"),
            )
            weights = F.dropout(weights, p=self.dropout.p, training=self.training)
            output = torch.matmul(weights, xv)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(cast_floating_to_module_dtype(output, self.wo))
        if trace is not None and trace.requires(f"{site_prefix}.update"):
            output = trace.emit(
                f"{site_prefix}.update",
                output,
                axes=("batch", "time", "channel"),
                width=self.dim_model,
            )
        return output


class SequifierEncoderLayer(nn.Module):
    @beartype
    def __init__(self, architecture):
        super().__init__()
        dim_model = architecture.dim_model
        self.dim_model = dim_model
        self.norm_first = architecture.normalization.norm_first

        # Normalization
        normalization_type = architecture.normalization.type
        NormClass = RMSNorm if normalization_type == "rmsnorm" else nn.LayerNorm
        norm_eps = 1e-6 if normalization_type == "rmsnorm" else 1e-3
        self.norm1 = NormClass(dim_model, eps=norm_eps)
        self.norm2 = NormClass(dim_model, eps=norm_eps)

        # Attention
        self.attn = SelfAttention(
            dim_model=dim_model,
            n_heads=architecture.attention.n_heads,
            n_kv_heads=architecture.attention.n_kv_heads,
            attention_type=architecture.attention.type,
            dropout=architecture.dropout,
            context_length=architecture.max_context_length,
            output_projection=architecture.attention.output_projection,
            use_rope=(architecture.position_encoding.type == "rope"),
            rope_theta=architecture.position_encoding.theta,
        )

        # Feed Forward
        self.ff = FeedForward(
            dim_model,
            architecture.feed_forward.dim,
            architecture.feed_forward.activation,
            architecture.dropout,
        )
        self.dropout = nn.Dropout(architecture.dropout)

    @staticmethod
    @conditional_beartype
    def _residual_add(residual, update):
        if (
            residual.is_floating_point()
            and update.is_floating_point()
            and residual.dtype != update.dtype
        ):
            residual = residual.to(dtype=update.dtype)
        return residual + update

    @conditional_beartype
    def forward(
        self,
        src,
        src_mask=None,
        *,
        trace: TraceContext | None = None,
        site_prefix: str = "",
    ):
        # Pre-LN vs Post-LN logic
        if self.norm_first:
            normed_src = self.norm1(cast_floating_to_module_dtype(src, self.norm1))
            if trace is not None:
                normed_src = trace.emit(
                    f"{site_prefix}.attention.norm_input",
                    normed_src,
                    axes=("batch", "time", "channel"),
                    width=self.dim_model,
                )
            attention_update = self.dropout(
                self.attn(
                    normed_src,
                    mask=src_mask,
                    trace=trace,
                    site_prefix=f"{site_prefix}.attention",
                )
            )
            x = self._residual_add(
                src,
                attention_update,
            )
            if trace is not None:
                x = trace.emit(
                    f"{site_prefix}.attention.output",
                    x,
                    axes=("batch", "time", "channel"),
                    width=self.dim_model,
                )
            normed_x = self.norm2(cast_floating_to_module_dtype(x, self.norm2))
            if trace is not None:
                normed_x = trace.emit(
                    f"{site_prefix}.mlp.norm_input",
                    normed_x,
                    axes=("batch", "time", "channel"),
                    width=self.dim_model,
                )
            mlp_update = self.dropout(
                self.ff(normed_x, trace=trace, site_prefix=f"{site_prefix}.mlp")
            )
            if trace is not None:
                mlp_update = trace.emit(
                    f"{site_prefix}.mlp.update",
                    mlp_update,
                    axes=("batch", "time", "channel"),
                    width=self.dim_model,
                )
            x = self._residual_add(x, mlp_update)
        else:
            attention_input = src
            if trace is not None:
                attention_input = trace.emit(
                    f"{site_prefix}.attention.norm_input",
                    attention_input,
                    axes=("batch", "time", "channel"),
                    width=self.dim_model,
                )
            x = self._residual_add(
                src,
                self.dropout(
                    self.attn(
                        attention_input,
                        mask=src_mask,
                        trace=trace,
                        site_prefix=f"{site_prefix}.attention",
                    )
                ),
            )
            x = self.norm1(cast_floating_to_module_dtype(x, self.norm1))
            if trace is not None:
                x = trace.emit(
                    f"{site_prefix}.attention.output",
                    x,
                    axes=("batch", "time", "channel"),
                    width=self.dim_model,
                )
                x = trace.emit(
                    f"{site_prefix}.mlp.norm_input",
                    x,
                    axes=("batch", "time", "channel"),
                    width=self.dim_model,
                )
            mlp_update = self.dropout(
                self.ff(x, trace=trace, site_prefix=f"{site_prefix}.mlp")
            )
            if trace is not None:
                mlp_update = trace.emit(
                    f"{site_prefix}.mlp.update",
                    mlp_update,
                    axes=("batch", "time", "channel"),
                    width=self.dim_model,
                )
            x = self._residual_add(x, mlp_update)
            x = self.norm2(cast_floating_to_module_dtype(x, self.norm2))
        return x
