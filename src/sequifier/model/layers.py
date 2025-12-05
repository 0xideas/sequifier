import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 1. Cast input to float32 once for stability
        x_fp32 = x.to(torch.float32)

        # 2. Calculate variance
        var = torch.mean(x_fp32.pow(2), dim=-1, keepdim=True)

        # 3. Normalize
        x_normed = x_fp32 * torch.rsqrt(var + self.eps)

        # 4. Cast back to the *input tensor's* dtype (traceable),
        #    rather than self.weight.dtype (not traceable in Cast ops)
        return self.weight * x_normed.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._update_cos_sin_cache(max_seq_len)

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

    def forward(self, x, seq_len):
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[
            :, :, :seq_len, ...
        ]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # Ensure cos/sin match q/k dtype (fix for Mixed Precision/ONNX)
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CustomFeedForward(nn.Module):
    def __init__(self, dim_model, dim_feedforward, activation_fn, dropout):
        super().__init__()
        self.activation_fn = activation_fn

        if activation_fn == "swiglu":
            # SwiGLU requires 2 gates, so we often adjust dim_feedforward or keep it
            # but implement the GLU split. Commonly SwiGLU hidden dim is 2/3 of standard.
            # Here we strictly follow config dim_feedforward.
            self.w1 = nn.Linear(dim_model, dim_feedforward)
            self.w2 = nn.Linear(dim_model, dim_feedforward)  # Gate
            self.w3 = nn.Linear(dim_feedforward, dim_model)  # Output
        else:
            self.linear1 = nn.Linear(dim_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, dim_model)
            self.act = nn.GELU() if activation_fn == "gelu" else nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.activation_fn == "swiglu":
            return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))
        else:
            return self.linear2(self.dropout(self.act(self.linear1(x))))


class CustomSelfAttention(nn.Module):
    def __init__(
        self,
        dim_model,
        n_head,
        n_kv_heads,
        attention_type,
        dropout,
        seq_length,
        use_rope=False,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_head
        self.dim_model = dim_model
        self.head_dim = dim_model // n_head
        self.attention_type = attention_type
        self.use_rope = use_rope

        self.wq = nn.Linear(dim_model, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(dim_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, dim_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryEmbedding(
                self.head_dim, max_seq_len=seq_length, theta=rope_theta
            )
            if self.head_dim % 2 != 0:
                raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        xq = (
            self.wq(x)
            .view(batch_size, seq_len, self.n_head, self.head_dim)
            .transpose(1, 2)
        )
        xk = (
            self.wk(x)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        xv = (
            self.wv(x)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.use_rope:
            cos, sin = self.rope(xv, seq_len)
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # Handle GQA/MQA by repeating keys/values
        if self.n_kv_heads != self.n_head:
            n_rep = self.n_head // self.n_kv_heads
            xk = xk.repeat_interleave(n_rep, dim=1)
            xv = xv.repeat_interleave(n_rep, dim=1)

        # Scaled Dot Product Attention
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class SequifierEncoderLayer(nn.Module):
    def __init__(self, config, dim_model, n_head, dim_feedforward, dropout, seq_length):
        super().__init__()
        self.norm_first = config.norm_first

        # Normalization
        NormClass = RMSNorm if config.normalization == "rmsnorm" else nn.LayerNorm
        self.norm1 = NormClass(dim_model)
        self.norm2 = NormClass(dim_model)

        # Attention
        self.attn = CustomSelfAttention(
            dim_model=dim_model,
            n_head=n_head,
            n_kv_heads=config.n_kv_heads,
            attention_type=config.attention_type,
            dropout=dropout,
            seq_length=seq_length,
            use_rope=(config.positional_encoding == "rope"),
            rope_theta=config.rope_theta,
        )

        # Feed Forward
        self.ff = CustomFeedForward(
            dim_model, dim_feedforward, config.activation_fn, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Pre-LN vs Post-LN logic
        if self.norm_first:
            x = src + self.dropout(self.attn(self.norm1(src), mask=src_mask))
            x = x + self.dropout(self.ff(self.norm2(x)))
        else:
            x = self.norm1(src + self.dropout(self.attn(src, mask=src_mask)))
            x = self.norm2(x + self.dropout(self.ff(x)))
        return x
