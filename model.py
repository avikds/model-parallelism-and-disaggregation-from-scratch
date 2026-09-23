"""
Model Parallelism and Disaggregation from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - CommLog
import torch

def tensor_bytes(t):
    """Return the number of bytes occupied by a tensor."""
    return t.numel() * t.element_size()


class CommLog:
    def __init__(self):
        self.events = []

    def record(self, op, n_devices, nbytes):
        """Record a communication event."""
        self.events.append((op, n_devices, nbytes))

    def total_bytes(self):
        """Return the total number of bytes recorded across all events."""
        return sum(event[2] for event in self.events)

    def count(self, op=None):
        """
        Return the number of recorded events.

        If op is None, count all events.
        Otherwise, count only events matching the given operation.
        """
        if op is None:
            return len(self.events)

        return sum(1 for event in self.events if event[0] == op)


def all_reduce(shards, log=None):
    """
    Elementwise-sum all device shards and return the same result on
    every device.
    """
    n = len(shards)

    # Elementwise sum across all devices.
    result = shards[0].clone()
    for shard in shards[1:]:
        result = result + shard

    # The collective communicates one shard's worth of bytes.
    if log is not None:
        log.record("all_reduce", n, tensor_bytes(shards[0]))

    # Every device receives the same reduced tensor.
    return [result.clone() for _ in range(n)]


def all_gather(shards, dim, log=None):
    """
    Concatenate all device shards along `dim` and return the complete
    tensor on every device.
    """
    n = len(shards)

    # Gather the shards along the requested dimension.
    result = torch.cat(shards, dim=dim)

    # Log the bytes of one input shard.
    if log is not None:
        log.record("all_gather", n, tensor_bytes(shards[0]))

    # Every device receives the complete gathered tensor.
    return [result.clone() for _ in range(n)]


def reduce_scatter(shards, dim, log=None):
    """
    Sum all shards elementwise, split the result into n chunks along
    `dim`, and return chunk i to device i.
    """
    n = len(shards)

    # First perform the elementwise reduction.
    reduced = shards[0].clone()
    for shard in shards[1:]:
        reduced = reduced + shard

    # Split the reduced tensor into n chunks along the requested dimension.
    chunks = torch.chunk(reduced, n, dim=dim)

    # Log the bytes of one input shard.
    if log is not None:
        log.record("reduce_scatter", n, tensor_bytes(shards[0]))

    # Device i receives the i-th chunk.
    return [chunks[i].clone() for i in range(n)]


def all_to_all(buckets, log=None):
    """
    buckets[i][j] is the tensor sent from device i to device j.

    Returns:
        received[j][i] = buckets[i][j]
    """
    n = len(buckets)

    # Construct the received buckets by transposing the source/destination
    # device indices.
    received = [
        [buckets[src][dst] for src in range(n)]
        for dst in range(n)
    ]

    # Count only inter-device traffic; local i -> i transfers are excluded.
    total_bytes = sum(
        tensor_bytes(buckets[i][j])
        for i in range(n)
        for j in range(n)
        if i != j
    )

    if log is not None:
        log.record("all_to_all", n, total_bytes)

    return received


def ring_time(nbytes, n, bandwidth, latency):
    """Return the communication time under the ring model."""
    return (
        2 * (n - 1) / n * nbytes / bandwidth
        + 2 * (n - 1) * latency
    )

# Step 2 - tp_mlp_forward
def init_block(d, n_heads, hidden, seed, scale=0.05):
    """
    Initialize the transformer block weights deterministically.
    """
    generator = torch.Generator().manual_seed(seed)

    return {
        "n_heads": n_heads,
        "wq": torch.randn(d, d, generator=generator, dtype=torch.float32) * scale,
        "wk": torch.randn(d, d, generator=generator, dtype=torch.float32) * scale,
        "wv": torch.randn(d, d, generator=generator, dtype=torch.float32) * scale,
        "wo": torch.randn(d, d, generator=generator, dtype=torch.float32) * scale,
        "w1": torch.randn(d, hidden, generator=generator, dtype=torch.float32) * scale,
        "w3": torch.randn(d, hidden, generator=generator, dtype=torch.float32) * scale,
        "w2": torch.randn(hidden, d, generator=generator, dtype=torch.float32) * scale,
        "g1": torch.ones(d, dtype=torch.float32),
        "g2": torch.ones(d, dtype=torch.float32),
    }


def rmsnorm(x, g, eps=1e-6):
    """
    Apply RMS normalization along the last dimension, then
    scale by the gain vector g.
    """
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * g


def mlp_forward(x, w):
    """
    Standard MLP forward pass.
    """
    return (torch.nn.functional.silu(x @ w["w1"]) * (x @ w["w3"])) @ w["w2"]


def shard_column(W, n):
    """
    Split W into n shards along the output-feature dimension (dim=1).
    """
    return list(torch.chunk(W, n, dim=1))


def shard_row(W, n):
    """
    Split W into n shards along the input-feature dimension (dim=0).
    """
    return list(torch.chunk(W, n, dim=0))


def tp_mlp_forward(x, w, n, log=None):
    """
    Tensor-parallel MLP forward pass.

    w1 and w3 are column-sharded.
    w2 is row-sharded.
    Each device computes a partial output, followed by one all-reduce.
    Device 0's reduced result is returned.
    """
    w1_shards = shard_column(w["w1"], n)
    w3_shards = shard_column(w["w3"], n)
    w2_shards = shard_row(w["w2"], n)

    partial_outputs = []

    for i in range(n):
        h1 = x @ w1_shards[i]
        h3 = x @ w3_shards[i]
        hidden = torch.nn.functional.silu(h1) * h3
        partial_outputs.append(hidden @ w2_shards[i])

    reduced = all_reduce(partial_outputs, log=log)

    return reduced[0]

# Step 3 - tp_attention_forward
import math

def attention_forward(x, w):
    """
    Standard causal multi-head self-attention.

    x: (B, T, d)
    """
    B, T, d = x.shape
    n_heads = w["n_heads"]
    head_dim = d // n_heads

    # Project inputs to queries, keys, and values.
    q = x @ w["wq"]
    k = x @ w["wk"]
    v = x @ w["wv"]

    # Split the hidden dimension into attention heads.
    q = q.reshape(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, T, n_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention.
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Causal mask: position t can only attend to positions <= t.
    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=x.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ v

    # Merge heads back into the hidden dimension.
    out = out.transpose(1, 2).reshape(B, T, d)

    # Final output projection.
    return out @ w["wo"]


def tp_attention_forward(x, w, n, log=None):
    """
    Tensor-parallel multi-head attention.

    Each device owns:
      - a column shard of wq, wk, and wv containing whole heads
      - a row shard of wo corresponding to those heads

    Each device computes a partial output and the partial outputs are
    all-reduced. Device 0's result is returned.
    """
    B, T, d = x.shape
    n_heads = w["n_heads"]
    head_dim = d // n_heads

    q_shards = shard_column(w["wq"], n)
    k_shards = shard_column(w["wk"], n)
    v_shards = shard_column(w["wv"], n)
    wo_shards = shard_row(w["wo"], n)

    heads_per_device = n_heads // n
    partial_outputs = []

    for i in range(n):
        local_dim = heads_per_device * head_dim

        # Project onto this device's local heads.
        q = x @ q_shards[i]
        k = x @ k_shards[i]
        v = x @ v_shards[i]

        # Arrange as (B, local_heads, T, head_dim).
        q = q.reshape(B, T, heads_per_device, head_dim).transpose(1, 2)
        k = k.reshape(B, T, heads_per_device, head_dim).transpose(1, 2)
        v = v.reshape(B, T, heads_per_device, head_dim).transpose(1, 2)

        # Scaled causal self-attention for the local heads.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        local_out = attn @ v

        # Merge this device's heads: (B, T, local_dim).
        local_out = local_out.transpose(1, 2).reshape(B, T, local_dim)

        # Apply this device's row shard of the output projection.
        partial_outputs.append(local_out @ wo_shards[i])

    # Combine contributions from all devices.
    reduced = all_reduce(partial_outputs, log=log)

    return reduced[0]


def block_forward(x, w):
    """
    Standard pre-norm transformer block:
        x + attention(rmsnorm(x, g1))
        + mlp(rmsnorm(previous_output, g2))
    """
    attn_input = rmsnorm(x, w["g1"])
    x = x + attention_forward(attn_input, w)

    mlp_input = rmsnorm(x, w["g2"])
    x = x + mlp_forward(mlp_input, w)

    return x


def tp_block_forward(x, w, n, log=None):
    """
    Tensor-parallel pre-norm transformer block.

    The attention and MLP each perform one all-reduce, giving exactly
    two all-reduces for the complete transformer layer.
    """
    attn_input = rmsnorm(x, w["g1"])
    x = x + tp_attention_forward(attn_input, w, n, log=log)

    mlp_input = rmsnorm(x, w["g2"])
    x = x + tp_mlp_forward(mlp_input, w, n, log=log)

    return x

# Step 4 - tp_layer_traffic
def tp_layer_traffic(batch, seq, d, bytes_per_elem):
    """
    Return the payload carried by one all-reduce in a transformer layer.
    """
    return batch * seq * d * bytes_per_elem


def tp_token_time(n_layers, batch, d, bytes_per_elem, n, bandwidth, latency):
    """
    Return tensor-parallel communication time per decode token.

    Each transformer layer performs two all-reduces.
    """
    if n == 1:
        return 0.0

    nbytes = batch * d * bytes_per_elem

    return 2 * n_layers * ring_time(
        nbytes,
        n,
        bandwidth,
        latency,
    )


def weight_bytes_per_device(total_weight_bytes, n):
    """
    Return the weight memory assigned to one tensor-parallel device.
    """
    return total_weight_bytes / n


def tp_report(
    n_layers,
    d,
    batch,
    total_weight_bytes,
    bytes_per_elem,
    hw,
    n_options,
):
    """
    Return tensor-parallel tradeoff information for each TP degree.
    """
    rows = []

    for n in n_options:
        weight_bytes = weight_bytes_per_device(total_weight_bytes, n)

        # Communication time for all TP all-reduces per decode token.
        comm_time = tp_token_time(
            n_layers,
            batch,
            d,
            bytes_per_elem,
            n,
            hw["bandwidth"],
            hw["latency"],
        )

        # Per-device memory-bound compute time uses HBM bandwidth.
        compute_time = weight_bytes / hw["hbm_bandwidth"]

        total_time = comm_time + compute_time

        comm_share = (
            comm_time / total_time
            if total_time != 0
            else 0.0
        )

        rows.append(
            {
                "tp": n,
                "weight_gb_per_device": round(weight_bytes / 1e9, 4),
                "comm_ms_per_token": round(comm_time * 1000, 4),
                "comm_share": round(comm_share, 4),
            }
        )

    return rows


def measured_matches_analytic(x, w, n):
    """
    Verify that the simulated TP block moves exactly the analytically
    expected number of bytes: two all-reduces, each carrying one
    batch * sequence * hidden-size payload.
    """
    B, T, d = x.shape

    log = CommLog()

    tp_block_forward(x, w, n, log=log)

    expected_bytes = 2 * tp_layer_traffic(
        B,
        T,
        d,
        x.element_size(),
    )

    return log.total_bytes() == expected_bytes

# Step 5 - pipeline_forward
def assign_stages(n_layers, n_stages):
    """
    Assign contiguous layer indices to pipeline stages.

    The first (n_layers % n_stages) stages receive one extra layer.
    """
    base_layers = n_layers // n_stages
    extra_layers = n_layers % n_stages

    stages = []
    current_layer = 0

    for stage in range(n_stages):
        stage_size = base_layers + (1 if stage < extra_layers else 0)

        stages.append(
            list(range(current_layer, current_layer + stage_size))
        )

        current_layer += stage_size

    return stages


def pipeline_bubble(n_stages, n_microbatches):
    """
    Return the idle fraction of a GPipe-style pipeline schedule.
    """
    return (n_stages - 1) / (n_microbatches + n_stages - 1)


def pipeline_makespan(n_stages, n_microbatches, stage_time):
    """
    Return the total pipeline makespan.
    """
    return (n_microbatches + n_stages - 1) * stage_time


def model_forward(x, blocks):
    """
    Apply all transformer blocks sequentially.
    """
    for block in blocks:
        x = block_forward(x, block)

    return x


def pipeline_forward(x, blocks, n_stages, log=None):
    """
    Run the model stage by stage.

    Each stage processes its assigned contiguous subset of blocks.
    After every stage except the final stage, record the activation
    transfer to the next pipeline stage.
    """
    stages = assign_stages(len(blocks), n_stages)

    for stage_idx, layer_indices in enumerate(stages):
        for layer_idx in layer_indices:
            x = block_forward(x, blocks[layer_idx])

        # The activation is sent to the next pipeline stage.
        if stage_idx < n_stages - 1 and log is not None:
            log.record("p2p", 2, tensor_bytes(x))

    return x


def pipeline_traffic(batch, seq, d, bytes_per_elem, n_stages):
    """
    Return the total activation traffic between pipeline stages.
    """
    return (
        (n_stages - 1)
        * batch
        * seq
        * d
        * bytes_per_elem
    )

# Step 6 - expert_parallel_dispatch
import torch


def init_experts(n_experts, d, hidden, seed):
    """
    Initialize all experts from a single deterministically seeded generator.

    For each expert, weights are generated in the order:
    w1, w3, w2.
    """
    generator = torch.Generator().manual_seed(seed)

    experts = []

    for _ in range(n_experts):
        w1 = torch.randn(
            d,
            hidden,
            generator=generator,
            dtype=torch.float32,
        ) * 0.05

        w3 = torch.randn(
            d,
            hidden,
            generator=generator,
            dtype=torch.float32,
        ) * 0.05

        w2 = torch.randn(
            hidden,
            d,
            generator=generator,
            dtype=torch.float32,
        ) * 0.05

        experts.append({
            "w1": w1,
            "w3": w3,
            "w2": w2,
        })

    return experts


def expert_forward(x, e):
    """
    Gated MLP forward pass for one expert.
    """
    return (torch.nn.functional.silu(x @ e["w1"]) * (x @ e["w3"])) @ e["w2"]


def moe_dense_forward(x_tokens, expert_ids, experts):
    """
    Reference MoE implementation where every token is evaluated
    directly by its assigned expert.
    """
    out = torch.empty_like(x_tokens)

    for i in range(x_tokens.shape[0]):
        expert_id = int(expert_ids[i].item())
        out[i] = expert_forward(
            x_tokens[i:i + 1],
            experts[expert_id],
        )[0]

    return out


def expert_parallel_dispatch(x_tokens, expert_ids, experts, n_devices, log=None):
    """
    Route tokens to the devices containing their experts, execute the
    experts, and route the outputs back to the original token owners.

    Returns:
        out: results in the original token order
        off_device_tokens: number of tokens whose expert is on another device
    """
    n_tokens, d = x_tokens.shape
    n_experts = len(experts)

    # Each device hosts an equal number of experts.
    experts_per_device = n_experts // n_devices

    # Initial token ownership is contiguous.
    token_chunks = torch.arange(
        n_tokens,
        device=x_tokens.device,
    ).chunk(n_devices)

    # buckets[src][dst] contains tokens sent from src to dst.
    buckets = [
        [None for _ in range(n_devices)]
        for _ in range(n_devices)
    ]

    # Keep token indices and expert IDs as simulation metadata.
    bucket_indices = [
        [[] for _ in range(n_devices)]
        for _ in range(n_devices)
    ]

    bucket_experts = [
        [[] for _ in range(n_devices)]
        for _ in range(n_devices)
    ]

    off_device_tokens = 0

    for src in range(n_devices):
        for idx_tensor in token_chunks[src]:
            token_idx = int(idx_tensor.item())
            expert_id = int(expert_ids[token_idx].item())

            dst = expert_id // experts_per_device

            bucket_indices[src][dst].append(token_idx)
            bucket_experts[src][dst].append(expert_id)

            if src != dst:
                off_device_tokens += 1

    # Construct tensor buckets for the first all-to-all.
    for src in range(n_devices):
        for dst in range(n_devices):
            indices = bucket_indices[src][dst]

            if indices:
                idx = torch.tensor(
                    indices,
                    dtype=torch.long,
                    device=x_tokens.device,
                )
                buckets[src][dst] = x_tokens[idx]
            else:
                buckets[src][dst] = x_tokens.new_empty((0, d))

    # First exchange: owning devices -> expert devices.
    received = all_to_all(buckets, log=log)

    # Compute each received token using its assigned expert.
    return_buckets = [
        [None for _ in range(n_devices)]
        for _ in range(n_devices)
    ]

    for dst in range(n_devices):
        for src in range(n_devices):
            tokens = received[dst][src]
            expert_list = bucket_experts[src][dst]

            if tokens.shape[0] == 0:
                return_buckets[dst][src] = tokens.new_empty((0, d))
                continue

            result = torch.empty_like(tokens)

            for j, expert_id in enumerate(expert_list):
                result[j] = expert_forward(
                    tokens[j:j + 1],
                    experts[expert_id],
                )[0]

            # Results are sent back to their original owner.
            return_buckets[dst][src] = result

    # Second exchange: expert devices -> original owning devices.
    returned = all_to_all(return_buckets, log=log)

    # Reassemble results in the original token order.
    out = torch.empty_like(x_tokens)

    for src in range(n_devices):
        for dst in range(n_devices):
            indices = bucket_indices[src][dst]

            if indices:
                idx = torch.tensor(
                    indices,
                    dtype=torch.long,
                    device=x_tokens.device,
                )
                out[idx] = returned[src][dst]

    return out, off_device_tokens


def load_imbalance(expert_ids, n_experts):
    """
    Return max expert load divided by mean expert load.
    """
    counts = torch.bincount(
        expert_ids.to(torch.long),
        minlength=n_experts,
    ).to(torch.float32)

    mean_load = counts.mean()

    if mean_load == 0:
        return 0.0

    return float(counts.max() / mean_load)

# Step 7 - strategy_report
def device_memory(weight_bytes, kv_bytes, tp, pp):
    """
    Return the per-device memory footprint for a given TP x PP split.
    """
    return (weight_bytes + kv_bytes) / (tp * pp)


def token_time(
    weight_bytes,
    kv_bytes,
    n_layers,
    batch,
    d,
    bytes_per_elem,
    tp,
    pp,
    hw,
):
    """
    Estimate the time to process one decode token.

    The total consists of:
      1. Memory-bound weight + KV streaming time.
      2. Tensor-parallel communication time.
      3. Point-to-point communication between pipeline stages.
    """
    # Per-device bytes streamed for one decode token.
    streamed_bytes = (weight_bytes + kv_bytes) / tp

    memory_time = streamed_bytes / hw["hbm_bandwidth"]

    # Tensor-parallel communication.
    comm_time = tp_token_time(
        n_layers,
        batch,
        d,
        bytes_per_elem,
        tp,
        hw["bandwidth"],
        hw["latency"],
    )

    # Point-to-point transfers between consecutive pipeline stages.
    p2p_time = (pp - 1) * (
        batch * d * bytes_per_elem / hw["bandwidth"]
        + hw["latency"]
    )

    return memory_time + comm_time + p2p_time


def strategy_report(
    weight_bytes,
    kv_bytes,
    n_layers,
    d,
    batch,
    bytes_per_elem,
    n_gpus,
    hw,
):
    """
    Enumerate all tensor-parallel x pipeline-parallel configurations
    whose product equals n_gpus, with both dimensions powers of two.
    """
    rows = []

    for tp in range(1, n_gpus + 1):
        # Both TP and PP must be powers of two.
        if tp & (tp - 1):
            continue

        if n_gpus % tp != 0:
            continue

        pp = n_gpus // tp

        if pp & (pp - 1):
            continue

        memory = device_memory(
            weight_bytes,
            kv_bytes,
            tp,
            pp,
        )

        ms_per_token = token_time(
            weight_bytes,
            kv_bytes,
            n_layers,
            batch,
            d,
            bytes_per_elem,
            tp,
            pp,
            hw,
        ) * 1000

        rows.append(
            {
                "tp": tp,
                "pp": pp,
                "memory_gb": round(memory / 1e9, 3),
                "fits": memory <= hw["memory_bytes"],
                "ms_per_token": round(ms_per_token, 3),
            }
        )

    return rows


def best_strategy(report):
    """
    Return the lowest-latency strategy among configurations that fit
    within the available device memory.

    Return None when no configuration fits.
    """
    fitting = [row for row in report if row["fits"]]

    if not fitting:
        return None

    return min(fitting, key=lambda row: row["ms_per_token"])

# Step 8 - disaggregation_report
def kv_transfer_time(prompt_len, kv_bytes_per_token, link_bandwidth):
    """
    Return the time required to transfer the prompt's KV cache.
    """
    return (prompt_len * kv_bytes_per_token) / link_bandwidth


def colocated_timeline(decode_step_s, n_steps, prefill_s, arrive_step):
    """
    Model a decode stream sharing a GPU with a newly arriving prefill.

    The prefill runs before the decode step at `arrive_step`, creating
    one inter-token latency spike. The new request's first token is
    produced by the following decode step.
    """
    itl_list = [decode_step_s] * n_steps

    if 0 <= arrive_step < n_steps:
        itl_list[arrive_step] = prefill_s + decode_step_s

    ttft_new = prefill_s + decode_step_s

    return (
        [round(t, 6) for t in itl_list],
        round(ttft_new, 6),
    )


def disaggregated_timeline(decode_step_s, n_steps, prefill_s, transfer_s):
    """
    Model decode running on a separate GPU pool from prefill.

    The existing decode stream is unaffected by the prefill.
    """
    itl_list = [round(decode_step_s, 6)] * n_steps

    ttft_new = prefill_s + transfer_s + decode_step_s

    return (
        itl_list,
        round(ttft_new, 6),
    )


def pool_sizes(arrival_rate, in_len, out_len, prefill_tps, decode_tps):
    """
    Return the required prefill and decode pool sizes.
    """
    prefill_load = arrival_rate * in_len / prefill_tps
    decode_load = arrival_rate * out_len / decode_tps

    # Inputs are non-negative in the intended workload model.
    prefill_pool = int(prefill_load) + (prefill_load > int(prefill_load))
    decode_pool = int(decode_load) + (decode_load > int(decode_load))

    return prefill_pool, decode_pool


def disaggregation_report(
    decode_step_s,
    n_steps,
    prefill_s,
    arrive_step,
    transfer_s,
):
    """
    Compare colocated and disaggregated execution.
    """
    colocated_itl, colocated_ttft = colocated_timeline(
        decode_step_s,
        n_steps,
        prefill_s,
        arrive_step,
    )

    disaggregated_itl, disaggregated_ttft = disaggregated_timeline(
        decode_step_s,
        n_steps,
        prefill_s,
        transfer_s,
    )

    colocated_max_itl = max(colocated_itl)
    disaggregated_max_itl = max(disaggregated_itl)

    return {
        "colocated_max_itl": round(colocated_max_itl, 6),
        "disaggregated_max_itl": round(disaggregated_max_itl, 6),
        "colocated_ttft": round(colocated_ttft, 6),
        "disaggregated_ttft": round(disaggregated_ttft, 6),
        "itl_spike": round(colocated_max_itl / decode_step_s, 3),
    }

