"""Bounded real-CUDA Phase 4 training benchmark; no checkpoints or run artifacts.

python -m scripts.benchmark_phase4_training --workers 0 2 4
Two exhausted, bounded iterators per split measure process respawn versus reuse.
"""
import argparse
import gc
import hashlib
import itertools
import json
import math
import multiprocessing
import platform
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader

from scripts.benchmark_phase4_data import record_batch, seed_cpu
from src.data.dataset import ROOT
from src.models.simple_cnn import SimpleCNN
from src.training.config import TrainingConfig
from src.training.data import build_loaders
from src.training.losses import make_criterion

MIB = 1024**2
GIB = 1024**3


class LimitedBatchSampler:
    """Keep the real sampler and exhaust a bounded iterator without image caching."""

    def __init__(self, source, batches):
        if batches < 1 or len(source.sampler) < batches * source.batch_size:
            raise ValueError("Insufficient full batches for the bounded benchmark")
        self.source, self.batches = source, batches

    def __iter__(self):
        return itertools.islice(iter(self.source), self.batches)

    def __len__(self):
        return self.batches


def bounded_loader(source, *, batches, persistent, seed):
    if persistent and source.num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    return DataLoader(
        source.dataset, batch_sampler=LimitedBatchSampler(source.batch_sampler, batches),
        num_workers=source.num_workers, persistent_workers=persistent,
        pin_memory=source.pin_memory, prefetch_factor=source.prefetch_factor,
        collate_fn=source.collate_fn,
        # Separate worker base seeds from the original sampler's generator.
        # Persistent iterator resets consume base seeds differently from respawn.
        generator=torch.Generator().manual_seed(seed),
    )


class HostMemoryMonitor:
    """Sample process-tree memory and available host RAM without per-step queries."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.peak_rss = self.peak_private = self.peak_children = 0
        self.min_available = psutil.virtual_memory().available
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        parent = psutil.Process()
        children = parent.children(recursive=True)
        rss = private = 0
        for process in [parent, *children]:
            try:
                info = process.memory_info()
                rss += info.rss
                private += getattr(info, "private", info.rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_private = max(self.peak_private, private)
        self.peak_children = max(self.peak_children, len(children))
        self.min_available = min(self.min_available, psutil.virtual_memory().available)

    def _run(self):
        try:
            while not self.stop_event.wait(0.25):
                self.sample()
        except Exception as exc:
            self.error = repr(exc)

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.sample()


@dataclass
class PassResult:
    split: str
    epoch: int
    workers: int
    persistent: bool
    pairs: int
    batches: int
    startup_s: float
    iterator_s: float
    first_wait_s: float
    end_s: float
    e2e_s: float
    wait_s: float
    model_wall_s: float
    h2d_gpu_s: float
    compute_gpu_s: float
    actual_optimizer_steps: int
    scale_before_measurement: float
    scale_after_measurement: float
    peak_allocated: int
    peak_reserved: int
    order_sha256: str
    worker_pids: list[int]

    @property
    def pairs_per_second(self):
        return self.pairs / self.e2e_s


def training_step(model, batch, criterion, optimizer, scaler, *, training):
    """Mirror run_epoch operations, including its finite-loss and metric transfers.

    Events separate H2D and GPU work without adding phase synchronizations.
    Wall model time includes H2D, host launch overhead and metric CPU transfers.
    """
    copy_start, copy_end, compute_end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    copy_start.record()
    pre, post, labels = (batch[key].to("cuda") for key in ("pre", "post", "label"))
    copy_end.record()
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        with torch.autocast(device_type="cuda", enabled=True):
            logits = model(pre, post)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite benchmark loss")
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        compute_end.record()
        # These synchronizing operations also occur in the real Phase 4 engine.
        mass = criterion.weight[labels].sum().item()
        loss_mass = loss.item() * mass
        targets = labels.detach().cpu().tolist()
        predictions = logits.detach().argmax(dim=1).cpu().tolist()
    return (copy_start, copy_end, compute_end), loss_mass, mass, targets, predictions


def measure_pass(loader, model, criterion, optimizer, scaler, *, epoch, split,
                 warmup, batches, seed):
    seed_cpu(seed)
    torch.cuda.manual_seed_all(seed)
    model.train(split == "train")
    digest = hashlib.sha256()
    pairs, steps = 0, [0]
    waits, walls, events = [], [], []
    targets, predictions = [], []
    loss_total = denominator = 0.0
    batch_size = loader.batch_sampler.source.batch_size
    iterator = None
    hook = optimizer.register_step_post_hook(lambda *args: steps.__setitem__(0, steps[0] + 1))
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        iterator = iter(loader)
        iterator_s = time.perf_counter() - started
        first_started = time.perf_counter()
        batch = next(iterator)
        first_wait_s = time.perf_counter() - first_started
        startup_s = time.perf_counter() - started
        processes = getattr(iterator, "_workers", ())
        pids = [p.pid for p in processes]
        if len(pids) != loader.num_workers or not all(p.is_alive() for p in processes):
            raise RuntimeError("Requested worker processes are not alive")
        for index in range(warmup):
            if index:
                batch = next(iterator)
            record_batch(batch, digest, batch_size, 128)
            training_step(model, batch, criterion, optimizer, scaler, training=split == "train")
            del batch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        steps[0] = 0
        scale_before = scaler.get_scale()
        started = time.perf_counter()
        for _ in range(batches):
            wait_started = time.perf_counter()
            batch = next(iterator)
            waits.append(time.perf_counter() - wait_started)
            model_started = time.perf_counter()
            event, loss_mass, mass, truth, pred = training_step(
                model, batch, criterion, optimizer, scaler, training=split == "train")
            walls.append(time.perf_counter() - model_started)
            events.append(event)
            loss_total += loss_mass
            denominator += mass
            targets.extend(truth)
            predictions.extend(pred)
            pairs += record_batch(batch, digest, batch_size, 128)
            del batch
        torch.cuda.synchronize()
        e2e_s = time.perf_counter() - started
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        end_started = time.perf_counter()
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("Bounded iterator did not exhaust")
        end_s = time.perf_counter() - end_started
        if split == "train" and steps[0] == 0:
            raise RuntimeError("AMP skipped every optimizer update; not a usable training measurement")
        return PassResult(
            split=split, epoch=epoch, workers=loader.num_workers,
            persistent=loader.persistent_workers, pairs=pairs, batches=batches,
            startup_s=startup_s, iterator_s=iterator_s, first_wait_s=first_wait_s, end_s=end_s,
            e2e_s=e2e_s, wait_s=sum(waits), model_wall_s=sum(walls),
            h2d_gpu_s=sum(a.elapsed_time(b) for a, b, c in events) / 1000,
            compute_gpu_s=sum(b.elapsed_time(c) for a, b, c in events) / 1000,
            actual_optimizer_steps=steps[0], scale_before_measurement=scale_before,
            scale_after_measurement=scaler.get_scale(),
            peak_allocated=allocated, peak_reserved=reserved,
            order_sha256=digest.hexdigest(), worker_pids=pids,
        )
    finally:
        hook.remove()
        if not loader.persistent_workers:
            shutdown_iterator(iterator)


def shutdown_iterator(iterator):
    # PyTorch has no public close API for an early/failed multiprocessing iterator.
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


def estimate_split_seconds(result, split_pairs, batch_size):
    # Startup includes first next(), already represented in the steady-state
    # batch mean; subtract that mean wait once to avoid double counting it.
    return (math.ceil(split_pairs / batch_size) * result.e2e_s / result.batches
            + max(0.0, result.startup_s - result.wait_s / result.batches) + result.end_s)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def print_pass(r):
    print(f"RESULT {r.split} epoch={r.epoch} workers={r.workers} persistent={r.persistent}: "
          f"{r.pairs_per_second:.2f} pairs/s; e2e={r.e2e_s:.3f}s; "
          f"wait={r.wait_s / r.batches * 1000:.3f} ms/b; "
          f"model_wall={r.model_wall_s / r.batches * 1000:.3f} ms/b; "
          f"H2D_GPU={r.h2d_gpu_s / r.batches * 1000:.3f} ms/b; "
          f"compute_GPU={r.compute_gpu_s / r.batches * 1000:.3f} ms/b; "
          f"startup={r.startup_s:.3f}s (iter={r.iterator_s:.3f}, first_wait={r.first_wait_s:.3f}); "
          f"end={r.end_s:.3f}s; optimizer_updates={r.actual_optimizer_steps}; "
          f"AMP_scale={r.scale_before_measurement:g}->{r.scale_after_measurement:g}; "
          f"peak_alloc/res={r.peak_allocated / MIB:.1f}/{r.peak_reserved / MIB:.1f} MiB; "
          f"pids={r.worker_pids}; order={r.order_sha256}", flush=True)


def print_summary(configurations, counts, batch_size):
    print("\nMEASURED CUDA TRAINING (epoch 2 steady-state throughput; no model selection)")
    print(f"{'W':>2} {'Persist':>7} {'Train/s':>9} {'Val/s':>9} {'Train min':>10} "
          f"{'Val min':>9} {'Total1 min':>10} {'Total2 min':>10} {'Start1 s':>9} "
          f"{'Start2 s':>9} {'GPU A/R MiB':>16} {'Host private GiB':>16} {'Min free GiB':>12}")
    for config in configurations:
        passes = config["passes"]
        t1, v1, t2, v2 = passes
        estimates = [estimate_split_seconds(r, counts[r.split], batch_size) for r in passes]
        print(f"{t2.workers:>2} {str(t2.persistent):>7} {t2.pairs_per_second:>9.1f} "
              f"{v2.pairs_per_second:>9.1f} {estimates[2] / 60:>10.2f} {estimates[3] / 60:>9.2f} "
              f"{sum(estimates[:2]) / 60:>10.2f} {sum(estimates[2:]) / 60:>10.2f} "
              f"{t1.startup_s + v1.startup_s:>9.2f} {t2.startup_s + v2.startup_s:>9.2f} "
              f"{max(r.peak_allocated for r in passes) / MIB:>7.1f}/"
              f"{max(r.peak_reserved for r in passes) / MIB:<7.1f} "
              f"{config['host_peak_private'] / GIB:>16.2f} {config['host_min_available'] / GIB:>12.2f}")
    print("Startup = iter(loader) through first CPU batch, including Windows spawn/import/serialization; "
          "epoch 2 uses the same PIDs only when persistence is enabled.")
    print("E2E = full wall window including next(), H2D, AMP forward/loss, backward, AdamW, "
          "engine-style metric transfers and benchmark bookkeeping; CUDA synchronized at both boundaries.")
    print("GPU event phases partition H2D and step work; model wall includes both plus host overhead. "
          "Do not add GPU-event times to model wall or E2E.")
    print("Estimate per split = ceil(full_pairs/batch) * measured E2E/batches + "
          "max(0, startup - mean_wait) + exhausted-iterator teardown. Total = train + validation.")
    print("Estimates exclude final sklearn metrics, logging, checkpoint I/O, scheduler and early stopping. "
          "Short-window sampling, disk cache, thermal/power state, partial final batch, AMP evolution "
          "and host paging introduce uncertainty; these are NOT measured full epochs.")
    print("AMP calls scaler.step() on every train batch; GradScaler can legitimately skip updates "
          "on gradient overflow, as in the real engine. Actual optimizer updates/scales are reported "
          "per pass; throughput counts processed pairs, not successful updates.")
    print("Host private memory is sampled process-tree private commit on Windows; RSS can double-count "
          "shared pages. Host peaks/minima sampled every 250 ms can miss brief spikes. "
          "CUDA allocated/reserved exclude driver/context and other applications.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--persistent", nargs="+", choices=["false", "true"], default=["false", "true"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--val-warmup-batches", type=int, default=5)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/patches_full/patches.csv")
    parser.add_argument("--json-output", type=Path, help="Optional small NEW JSON file; never overwrite")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.batches, args.warmup_batches, args.val_batches, args.val_warmup_batches) < 1:
        parser.error("Batch counts and batch size must be positive")
    if any(w < 0 for w in args.workers) or len(set(args.workers)) != len(args.workers):
        parser.error("Workers must be distinct nonnegative integers")
    if len(set(args.persistent)) != len(args.persistent):
        parser.error("Persistent modes must be distinct")
    if args.workers == [0] and args.persistent == ["true"]:
        parser.error("No valid configuration: persistence requires nonzero workers")
    if args.json_output is not None:
        output = args.json_output.resolve()
        protected = (ROOT / "results/phase4_cnn_weighted_ce_seed42").resolve()
        if output == protected or protected in output.parents or output.exists():
            parser.error("JSON output must be a new file outside the protected experiment")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback cannot measure GPU training throughput")
    device = torch.cuda.get_device_properties(0)
    seed_cpu(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    initial_state = {key: value.clone() for key, value in SimpleCNN().state_dict().items()}
    initial_sha = state_digest(initial_state)
    print(f"Python={platform.python_version()} torch={torch.__version__} GPU={device.name} "
          f"VRAM={device.total_memory / MIB:.0f} MiB; CPU threads={torch.get_num_threads()}; "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}; TF32={torch.backends.cudnn.allow_tf32}", flush=True)
    print(f"seed={args.seed}; batch={args.batch_size}; input=128; AdamW lr=0.0003 wd=0.0001; "
          f"Weighted CE from full train labels; CUDA AMP float16; init_SHA256={initial_sha}", flush=True)
    print(f"Two bounded epochs per configuration: train {args.warmup_batches}+{args.batches}, "
          f"val {args.val_warmup_batches}+{args.val_batches} warmup+measured. "
          "Default loader pin_memory=False/prefetch=2 retained. No test patches, checkpoints or full training.", flush=True)
    print("Sample identities/order checked across configurations per split/epoch. Augmentation settings "
          "match training, but worker-local RNG draws differ across counts/persistence. "
          "Disk cache is not cleared; run order and laptop power/thermal state can bias comparisons.", flush=True)
    reference, configurations = {}, []
    counts = None
    for workers in args.workers:
        for mode in args.persistent:
            persistent = mode == "true"
            if workers == 0 and persistent:
                continue
            print(f"\nCONFIG workers={workers} persistent={persistent}", flush=True)
            train_loader = val_loader = model = optimizer = criterion = scaler = data = None
            passes = []
            gc.collect()
            torch.cuda.empty_cache()
            with HostMemoryMonitor() as host:
                try:
                    config = TrainingConfig("simple_cnn", seed=args.seed, batch_size=args.batch_size,
                                            num_workers=workers)
                    data = build_loaders(config, args.manifest, project_root=ROOT)
                    counts = data.split_counts
                    print(f"Full split pairs={counts}; class_weights={data.class_weights.tolist()}", flush=True)
                    train_loader = bounded_loader(data.train, batches=args.warmup_batches + args.batches,
                                                  persistent=persistent, seed=args.seed)
                    val_loader = bounded_loader(data.val, batches=args.val_warmup_batches + args.val_batches,
                                                persistent=persistent, seed=args.seed)
                    model = SimpleCNN()
                    model.load_state_dict(initial_state)
                    if state_digest(model.state_dict()) != initial_sha:
                        raise RuntimeError("Initial weights differ")
                    model.cuda()
                    criterion = make_criterion("weighted_ce", weights=data.class_weights).cuda()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                                  weight_decay=config.weight_decay)
                    scaler = torch.amp.GradScaler("cuda", enabled=True)
                    for epoch in (1, 2):
                        # The original train generator belongs only to the sampler;
                        # independent loader generators keep sample order invariant.
                        data.train.generator.manual_seed(args.seed + epoch - 1)
                        for split, loader, warmup, batches in (
                            ("train", train_loader, args.warmup_batches, args.batches),
                            ("val", val_loader, args.val_warmup_batches, args.val_batches),
                        ):
                            print(f"Measuring {split} epoch={epoch} ...", flush=True)
                            result = measure_pass(loader, model, criterion, optimizer, scaler,
                                                  epoch=epoch, split=split, warmup=warmup,
                                                  batches=batches, seed=args.seed + epoch - 1)
                            key = (split, epoch)
                            if result.order_sha256 != reference.setdefault(key, result.order_sha256):
                                raise RuntimeError(f"Sample order differs for {key}")
                            if persistent and epoch == 2:
                                previous = next(r for r in passes if r.split == split)
                                if result.worker_pids != previous.worker_pids:
                                    raise RuntimeError("Persistent worker PIDs were not reused")
                            passes.append(result)
                            print_pass(result)
                    host.sample()  # Both persistent pools still alive here.
                finally:
                    for loader in (train_loader, val_loader):
                        shutdown_iterator(getattr(loader, "_iterator", None))
                    del loader
                    model = optimizer = criterion = scaler = data = train_loader = val_loader = None
                    gc.collect()
                    torch.cuda.empty_cache()
            if host.error:
                raise RuntimeError(f"Host memory monitor failed: {host.error}")
            entry = dict(passes=passes, host_peak_rss=host.peak_rss,
                         host_peak_private=host.peak_private, host_min_available=host.min_available,
                         peak_children=host.peak_children)
            configurations.append(entry)
            print(f"HOST peak private={host.peak_private / GIB:.2f} GiB; "
                  f"peak tree RSS={host.peak_rss / GIB:.2f} GiB; "
                  f"minimum system available={host.min_available / GIB:.2f} GiB; "
                  f"peak worker children={host.peak_children}", flush=True)
    print_summary(configurations, counts, args.batch_size)
    if args.json_output is not None:
        payload = dict(schema_version=1, settings={**vars(args), "manifest": str(args.manifest),
                       "json_output": str(args.json_output)}, gpu=device.name,
                       vram_bytes=device.total_memory, python=platform.python_version(),
                       torch=str(torch.__version__), initial_weights_sha256=initial_sha,
                       split_counts=counts, configurations=[
                           {**c, "passes": [asdict(r) for r in c["passes"]]} for c in configurations])
        with args.json_output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        print(f"Saved small results JSON: {args.json_output}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
