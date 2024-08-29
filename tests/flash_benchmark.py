import torch
import numpy as np
from tqdm import tqdm
from flash_attn import flash_attn_func

from matplotlib import pyplot as plt

print("Testing forward...")

def slow_attn_func(
    query, key, value, scale=None, dropout_p=0.0, causal=True, window_size=(-1, -1),
    alibi_slopes=None, deterministic=False):
    """
    Computes the scaled dot product attention between query, key, and value tensors in PyTorch eager mode.

    Args:
        query (torch.Tensor): The query tensor of shape (batch_size, n_heads, seq_len, hidden_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, n_heads, seq_len, hidden_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, n_heads, seq_len, hidden_dim).
        attn_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, n_heads, seq_len, seq_len). Defaults to None.
        is_causal (bool, optional): Whether to apply a causal attention mask. Defaults to False.
        dropout_p (float, optional): The dropout probability. Defaults to 0.
        scale (float, optional): The scale factor for the dot product. Defaults to None.

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, n_heads, seq_len, hidden_dim).
    """

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Calculate the scale factor
    scale_factor = 1 / np.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = (query @ key.transpose(-2, -1) * scale_factor)
    
    # Create the attention mask
    attn_mask = torch.ones(query.shape[0], query.shape[1], query.shape[2], query.shape[2], dtype=torch.bool, device=device).tril(diagonal=0) if causal else attn_mask
    attn_weight = attn_weight.masked_fill_(~attn_mask, -torch.inf) if attn_mask is not None else attn_weight
      
    # Compute the scaled dot product attention
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)

    out = attn_weight @ value

    return out.transpose(1, 2)

batch_size = 1
seq_len = 64
num_heads = 32
embed_dim = 128
dtype = torch.float16
device = torch.device("cuda")

query = torch.nn.Parameter(torch.rand(batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype))
key = torch.rand(batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype)
value = torch.rand(batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype)

eager = slow_attn_func(query, key, value, causal=True)
flash = flash_attn_func(query, key, value, causal=True)
print((eager - flash).norm() / flash.norm())
assert torch.allclose(eager, flash, rtol=1e-03,atol=1e-03)
print("Forward works.")

print("Testing backward...")

loss = flash.norm()
loss.backward()

print("Backward works.")


print("Testing speedup...")
def bench_attention(seq_len, flash=False, num_repeats=256):
    """
    Measures the average time (in ms) required to compute multi-head attention for sequences of a given length.

    Args:
        seq_len (int): The length of the input sequence.
        flash (bool, optional): Whether to use the FlashAttention algorithm. Defaults to False.
        num_repeats (int, optional): The number of times to repeat the attention computation for timing purposes. Defaults to 256.

    Returns:
        float: The average time (in ms) required to compute multi-head attention for sequences of length seq_len.
    """
    
    if flash:
        mha = flash_attn_func
    else:
        mha = slow_attn_func
        
    query = torch.nn.Parameter(torch.rand(batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype))
    key = torch.rand(batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype)
    value = torch.rand(batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(4):
        _ = mha(query, key, value, causal=True)

    start.record()
    for _ in range(num_repeats):
        _ = mha(query, key, value, causal=True)   
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / num_repeats

context_len = np.arange(256,4096,64)
flash = np.zeros(context_len.shape)
standard = np.zeros(context_len.shape)

for i,l in enumerate(tqdm(context_len)):
    flash[i] = bench_attention(l,flash=True)
    standard[i] = bench_attention(l,flash=False)

print("Average speedup : ", np.mean(standard / flash))

plt.plot(context_len, standard/flash)
plt.xlabel('Sequence length')
plt.ylabel('Speedup')
plt.title('Flash Attention vs. Standard Attention, head_size=128, n_heads=32, bs=1') 
plt.savefig("flash_v_vanilla.png", dpi=100, bbox_inches="tight")