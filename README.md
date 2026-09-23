# Model Parallelism and Disaggregation from Scratch

Chapters 5.4 and 5.5 of Inference Engineering, built on simulated devices so every byte a collective moves is counted. Implement all-reduce, all-gather, reduce-scatter and all-to-all over lists of tensors with a communication log and the ring-time model. Shard a transformer block's MLP by columns and rows and its attention by heads across devices, and verify that the tensor-parallel forward pass matches the single-device one to floating-point precision while logging exactly two all-reduces per layer. Assign layers to pipeline stages, measure the bubble, and route tokens to experts across devices with the all-to-all exchange that expert parallelism pays. Then decide: enumerate the tensor-by-pipeline splits of an eight-GPU node for a 70B model, price each one in memory per device and communication per token, and pick the fastest that fits. Finish with disaggregation, a timeline that shows a long prefill stalling co-located decode, the KV transfer that separating the two costs instead, and the pool sizes a given traffic mix needs.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** CommLog

---

Built on Deep-ML.
