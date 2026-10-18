"""CPU-only Phase 4 DataLoader benchmark; never imports the training engine.

Run from the project root with ``python -m scripts.benchmark_phase4_data``.
Only manifest metadata and bounded DataLoader queues are held in memory.
"""
import argparse
import hashlib
import json
import math
import multiprocessing
import platform
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import ROOT
from src.training.config import TrainingConfig
from src.training.data import build_loaders


@dataclass(frozen=True)
class Measurement:
    split: str
    workers: int
    samples: int
    batch_count: int
    mean_seconds: float
    median_seconds: float
    samples_per_second: float
    iterator_seconds: float
    first_batch_seconds: float
    startup_seconds: float
    estimated_loading_seconds: float
    order_digest: str


def seed_cpu(seed):
    # No CUDA initialization or training-side global configuration changes.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)


def record_batch(batch, digest, batch_size, input_size):
    count = len(batch["label"])
    expected = (count, 3, input_size, input_size)
    if count != batch_size or batch["pre"].shape != expected or batch["post"].shape != expected:
        raise ValueError("Benchmark requires full batches with Phase 4 input shapes")
    # Check actual returned identities, including warm-up, outside timed next().
    for image_id, uid in zip(batch["image_id"], batch["uid"], strict=True):
        digest.update(json.dumps([str(image_id), str(uid)]).encode("utf-8") + b"\n")
    return count


def measure_loader(loader, *, split, batches, warmup, seed, input_size=128):
    if batches < 1 or warmup < 1:
        raise ValueError("batches and warmup must be positive")
    if len(loader.dataset) < (warmup + batches) * loader.batch_size:
        raise ValueError(f"{split}: insufficient full batches for warm-up plus measurement")
    seed_cpu(seed)
    digest = hashlib.sha256()
    durations = []
    samples = 0
    iterator = None
    try:
        started = time.perf_counter()
        iterator = iter(loader)
        iterator_seconds = time.perf_counter() - started
        first_started = time.perf_counter()
        batch = next(iterator)
        first_batch_seconds = time.perf_counter() - first_started
        startup_seconds = time.perf_counter() - started
        record_batch(batch, digest, loader.batch_size, input_size)
        del batch
        if loader.num_workers:
            # Diagnostic only: private PyTorch worker handles are not used to
            # change scheduling, prefetching, or multiprocessing settings.
            processes = getattr(iterator, "_workers", ())
            if len(processes) != loader.num_workers or not all(p.is_alive() for p in processes):
                raise RuntimeError("Requested DataLoader worker processes are not alive")
        for _ in range(warmup - 1):
            batch = next(iterator)
            record_batch(batch, digest, loader.batch_size, input_size)
            del batch
        for _ in range(batches):
            started = time.perf_counter()
            batch = next(iterator)
            durations.append(time.perf_counter() - started)
            samples += record_batch(batch, digest, loader.batch_size, input_size)
            del batch
    finally:
        # Early bounded iteration must stop workers before the next setting.
        # PyTorch has no public iterator close API; keep this compatibility
        # access guarded and outside all reported timings.
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    mean = statistics.mean(durations)
    return Measurement(
        split=split, workers=loader.num_workers, samples=samples, batch_count=batches,
        mean_seconds=mean, median_seconds=statistics.median(durations),
        samples_per_second=samples / sum(durations), iterator_seconds=iterator_seconds,
        first_batch_seconds=first_batch_seconds, startup_seconds=startup_seconds,
        estimated_loading_seconds=mean * math.ceil(len(loader.dataset) / loader.batch_size),
        order_digest=digest.hexdigest(),
    )


def print_table(results):
    print("\nDATA LOADING ONLY - no GPU transfer, forward/backward, or optimizer step")
    print(f"{'Split':<7} {'Workers':>7} {'Batches':>7} {'Pairs':>7} {'Mean ms':>10} "
          f"{'Median ms':>10} {'Pairs/s':>10} {'Startup s':>10} {'Iter s':>9} "
          f"{'First s':>9} {'Est load s':>11}")
    for r in results:
        print(f"{r.split:<7} {r.workers:>7} {r.batch_count:>7} {r.samples:>7} "
              f"{r.mean_seconds * 1000:>10.3f} {r.median_seconds * 1000:>10.3f} "
              f"{r.samples_per_second:>10.1f} {r.startup_seconds:>10.3f} "
              f"{r.iterator_seconds:>9.3f} {r.first_batch_seconds:>9.3f} "
              f"{r.estimated_loading_seconds:>11.2f}")
    print("Startup = iterator creation through first delivered batch (also for workers=0); "
          "includes process spawn/import, dataset serialization, prefetch and first-batch work.")
    print("Iter = iter(loader); First = first next(iterator). Pure worker startup is not isolated.")
    print("Mean/median time only next(iterator) after warm-up; samples are PRE/POST pairs.")
    print("Est load = mean batch wait * ceil(split pairs / batch size), excluding startup. "
          "This is a data-only extrapolation, NOT actual full epoch time.")
    print("Manifest parsing, loader construction, identity checks and worker shutdown "
          "are outside the reported timings.")
    print("Tight-loop draining includes default prefetch/IPC behavior. Training compute can "
          "overlap loading, so these waits do not predict training stalls directly.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--warmup-batches", type=int, default=5,
                        help="Total warm-up batches, including the separately timed first batch")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/patches_full/patches.csv"))
    args = parser.parse_args(argv)
    if min(args.batch_size, args.batches, args.warmup_batches) < 1:
        parser.error("batch-size, batches and warmup-batches must be positive")
    if any(w < 0 for w in args.workers) or len(set(args.workers)) != len(args.workers):
        parser.error("workers must be distinct nonnegative integers")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(f"Python {platform.python_version()} | torch {torch.__version__} | "
          f"{platform.system()} | CPU threads={torch.get_num_threads()}", flush=True)
    print(f"seed={args.seed}; batch={args.batch_size}; warmup={args.warmup_batches}; "
          f"measured batches={args.batches}; worker order={args.workers}", flush=True)
    print("Using unmodified Phase 4 build_loaders (sampler=none, smoke=False, input=128). "
          "Train: paired flip/rotation; val: deterministic. Test patches are never opened.", flush=True)
    print("WARNING: OS disk cache and run order affect results; cache is not cleared. "
          "Repeat with reversed --workers order to assess bias. "
          "No full-dataset image cache; only bounded batch/prefetch buffers.", flush=True)
    print("Same seed and sample order per split; augmentation draws are reproducible per "
          "worker setting but need not be identical across worker counts.", flush=True)
    results, reference = [], {}
    for workers in args.workers:
        seed_cpu(args.seed)
        config = TrainingConfig("simple_cnn", batch_size=args.batch_size,
                                num_workers=workers, seed=args.seed)
        data = build_loaders(config, args.manifest, project_root=ROOT)
        print(f"workers={workers}; split pairs={data.split_counts}; "
              f"pin_memory={data.train.pin_memory}; prefetch_factor={data.train.prefetch_factor}; "
              f"persistent_workers={data.train.persistent_workers}", flush=True)
        for split, loader in (("train", data.train), ("val", data.val)):
            print(f"Measuring {split}, workers={workers} ...", flush=True)
            result = measure_loader(loader, split=split, batches=args.batches,
                                    warmup=args.warmup_batches, seed=args.seed)
            if result.order_digest != reference.setdefault(split, result.order_digest):
                raise RuntimeError(f"{split}: sample order differs across worker settings")
            results.append(result)
            print(f"  OK: {result.samples} measured pairs; "
                  f"order SHA256={result.order_digest}; worker processes verified={workers}", flush=True)
        del data, loader
    print_table(results)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
