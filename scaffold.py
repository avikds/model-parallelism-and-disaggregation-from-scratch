"""
Model Parallelism and Disaggregation from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Model Parallelism and Disaggregation from Scratch (Inference Engineering, chapters 5.4 and 5.5).

Story: collectives with a byte log; a transformer block sharded by columns, rows
and heads that matches the single device exactly; the two all-reduces per layer
priced against decode; pipeline stages and their bubble; expert routing over
all-to-all; the tensor-by-pipeline table for a 70B model on eight GPUs; and the
inter-token-latency spike that disaggregation removes.
"""
import torch


def main() -> None:
    torch.manual_seed(0)
    d, n_heads, hidden = 64, 8, 256
    blocks = [init_block(d, n_heads, hidden, seed=s) for s in range(4)]
    x = torch.randn(2, 8, d)

    # ---- 1. Tensor parallelism is exact ----
    log = CommLog()
    out = tp_block_forward(x, blocks[0], 4, log)
    print(f"tensor-parallel block over 4 devices: max |diff| vs single device {float((out - block_forward(x, blocks[0])).abs().max()):.2e}; "
          f"{log.count('all_reduce')} all-reduces of {log.events[0][2]} bytes each; analytic match {measured_matches_analytic(x, blocks[0], 4)}")
    hw = {"bandwidth": 300e9, "latency": 3e-6, "hbm_bandwidth": 3.35e12}
    print("70B model (140 GB BF16, 80 layers, d 8192), batch 8, NVLink 300 GB/s at 3 us per hop:")
    for r in tp_report(80, 8192, 8, 140e9, 2, hw, [1, 2, 4, 8]):
        print(f"  tp={r['tp']}: {r['weight_gb_per_device']:6.1f} GB per device, comm {r['comm_ms_per_token']:6.2f} ms per token ({r['comm_share']:.0%} of the step)")

    # ---- 2. Pipeline and expert parallelism ----
    log = CommLog()
    out = pipeline_forward(x, blocks, 2, log)
    print(f"\npipeline over 2 stages: exact {torch.allclose(out, model_forward(x, blocks), atol=1e-5)}, {log.count('p2p')} send of {log.events[0][2]} bytes; "
          f"bubble with 1 micro-batch {pipeline_bubble(4, 1):.0%}, with 16 {pipeline_bubble(4, 16):.0%}")
    experts = init_experts(8, d, hidden, seed=0)
    tokens = torch.randn(64, d)
    ids = torch.randint(0, 8, (64,))
    log = CommLog()
    out, off = expert_parallel_dispatch(tokens, ids, experts, 4, log)
    print(f"expert parallelism over 4 devices: exact {torch.allclose(out, moe_dense_forward(tokens, ids, experts), atol=1e-5)}, "
          f"{off} of 64 tokens routed off-device, {log.total_bytes()} bytes over two all-to-alls, load imbalance {load_imbalance(ids, 8):.2f}")

    # ---- 3. Choosing a split ----
    node = {"memory_bytes": 80e9, "hbm_bandwidth": 3.35e12, "bandwidth": 300e9, "latency": 3e-6}
    print("\n70B on eight 80 GB GPUs, decode batch 8:")
    rows = strategy_report(140e9, 10e9, 80, 8192, 8, 2, 8, node)
    for r in rows:
        print(f"  tp={r['tp']} pp={r['pp']}: {r['memory_gb']:5.1f} GB per device, fits {str(r['fits']):5s}, {r['ms_per_token']:6.2f} ms per token")
    b = best_strategy(rows)
    print(f"  best: tp={b['tp']} pp={b['pp']}")

    # ---- 4. Disaggregation ----
    transfer = kv_transfer_time(8192, 131072, 100e9)
    r = disaggregation_report(0.025, 40, 0.4, 10, transfer)
    print(f"\nco-located: a 400 ms prefill lands in the decode loop -> max ITL {r['colocated_max_itl'] * 1000:.0f} ms ({r['itl_spike']:.0f}x the step); "
          f"disaggregated: max ITL {r['disaggregated_max_itl'] * 1000:.0f} ms, TTFT {r['colocated_ttft'] * 1000:.0f} -> {r['disaggregated_ttft'] * 1000:.0f} ms after a {transfer * 1000:.1f} ms KV transfer")
    pre, dec = pool_sizes(50, 2000, 300, 20000, 2500)
    print(f"pools for 50 req/s at 2000 in / 300 out tokens: {pre} prefill replicas, {dec} decode replicas")


if __name__ == "__main__":
    main()

