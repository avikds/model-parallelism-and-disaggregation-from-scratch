# Model Parallelism and Disaggregation from Scratch

Chapters 5.4 and 5.5 of Inference Engineering, built on simulated devices so every byte a collective moves is counted. Implement all-reduce, all-gather, reduce-scatter and all-to-all over lists of tensors with a communication log and the ring-time model. Shard a transformer block's MLP by columns and rows and its attention by heads across devices, and verify that the tensor-parallel forward pass matches the single-device one to floating-point precision while logging exactly two all-reduces per layer. Assign layers to pipeline stages, measure the bubble, and route tokens to experts across devices with the all-to-all exchange that expert parallelism pays. Then decide: enumerate the tensor-by-pipeline splits of an eight-GPU node for a 70B model, price each one in memory per device and communication per token, and pick the fastest that fits. Finish with disaggregation, a timeline that shows a long prefill stalling co-located decode, the KV transfer that separating the two costs instead, and the pool sizes a given traffic mix needs.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** CommLog
- [x] **2.** tp_mlp_forward
- [x] **3.** tp_attention_forward
- [x] **4.** tp_layer_traffic
- [x] **5.** pipeline_forward
- [x] **6.** expert_parallel_dispatch
- [x] **7.** strategy_report
- [x] **8.** disaggregation_report

## Results

```
tensor-parallel block over 4 devices: max |diff| vs single device 2.38e-07; 2 all-reduces of 4096 bytes each; analytic match True
70B model (140 GB BF16, 80 layers, d 8192), batch 8, NVLink 300 GB/s at 3 us per hop:
  tp=1:  140.0 GB per device, comm   0.00 ms per token (0% of the step)
  tp=2:   70.0 GB per device, comm   1.03 ms per token (5% of the step)
  tp=4:   35.0 GB per device, comm   2.98 ms per token (22% of the step)
  tp=8:   17.5 GB per device, comm   6.84 ms per token (57% of the step)

pipeline over 2 stages: exact True, 1 send of 4096 bytes; bubble with 1 micro-batch 75%, with 16 16%
expert parallelism over 4 devices: exact True, 52 of 64 tokens routed off-device, 26624 bytes over two all-to-alls, load imbalance 1.50

70B on eight 80 GB GPUs, decode batch 8:
  tp=1 pp=8:  18.8 GB per device, fits True ,  44.80 ms per token
  tp=2 pp=4:  18.8 GB per device, fits True ,  23.43 ms per token
  tp=4 pp=2:  18.8 GB per device, fits True ,  14.18 ms per token
  tp=8 pp=1:  18.8 GB per device, fits True ,  12.44 ms per token
  best: tp=8 pp=1

co-located: a 400 ms prefill lands in the decode loop -> max ITL 425 ms (17x the step); disaggregated: max ITL 25 ms, TTFT 425 -> 436 ms after a 10.7 ms KV transfer
pools for 50 req/s at 2000 in / 300 out tokens: 5 prefill replicas, 6 decode replicas
```
