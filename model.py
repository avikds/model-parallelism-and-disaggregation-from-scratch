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

