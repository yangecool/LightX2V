import math
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from loguru import logger
from packaging.version import parse
from tqdm import tqdm

from lightx2v.utils.profiler import ExcludedProfilingContext
from lightx2v.utils.wan_runtune import get_wan_runtuner
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def resolve_offload_block_count(blocks_num, offload_ratio=1.0):
    """Return how many leading blocks should use streaming CPU offload."""
    try:
        ratio = float(offload_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"offload_ratio must be a number in [0, 1], got {offload_ratio!r}") from exc

    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"offload_ratio must be a finite number in [0, 1], got {offload_ratio!r}")
    if blocks_num < 0:
        raise ValueError(f"blocks_num must be non-negative, got {blocks_num}")

    # This preserves the historical `block_idx < ratio * blocks_num`
    # behavior when the product is not an integer.
    return min(blocks_num, math.ceil(ratio * blocks_num))


class WeightAsyncStreamManager(object):
    def __init__(self, offload_granularity):
        self.offload_granularity = offload_granularity
        self.init_stream = torch_device_module.Stream(priority=0)
        self.need_init_first_buffer = True
        self.lazy_load = False
        torch_version = parse(torch.__version__.split("+")[0])
        if AI_DEVICE == "cuda" and torch_version >= parse("2.7"):
            self.cuda_load_stream = torch_device_module.Stream(priority=1)
            self.compute_stream = torch_device_module.Stream(priority=1)
        else:
            self.cuda_load_stream = torch_device_module.Stream(priority=0)
            self.compute_stream = torch_device_module.Stream(priority=-1)

    def init_cpu_buffer(self, blocks_cpu_buffer=None, phases_cpu_buffer=None):
        self.need_init_first_buffer = True
        if self.offload_granularity == "block":
            assert blocks_cpu_buffer is not None
            self.cpu_buffers = [blocks_cpu_buffer[i] for i in range(len(blocks_cpu_buffer))]
        elif self.offload_granularity == "phase":
            assert phases_cpu_buffer is not None
            self.cpu_buffers = [phases_cpu_buffer[i] for i in range(len(phases_cpu_buffer))]
        else:
            raise NotImplementedError

    def init_cuda_buffer(self, blocks_cuda_buffer=None, phases_cuda_buffer=None):
        self.need_init_first_buffer = True
        if self.offload_granularity == "block":
            assert blocks_cuda_buffer is not None
            self.cuda_buffers = [blocks_cuda_buffer[i] for i in range(len(blocks_cuda_buffer))]
        elif self.offload_granularity == "phase":
            assert phases_cuda_buffer is not None
            self.cuda_buffers = [phases_cuda_buffer[i] for i in range(len(phases_cuda_buffer))]
        else:
            raise NotImplementedError

    def _sync(self):
        """Synchronize to ensure memory visibility across streams.

        XPU streams do not guarantee cross-stream memory visibility after a
        per-stream synchronize, so we use a device-wide sync on XPU.
        On CUDA, per-stream synchronize is sufficient and preferred.
        """
        tuner = get_wan_runtuner()
        started = time.perf_counter() if tuner.active else None
        if AI_DEVICE == "xpu":
            torch_device_module.synchronize()
        else:
            self.init_stream.synchronize()
        if started is not None:
            tuner.record_cpu_ms(
                "offload.init_wait",
                (time.perf_counter() - started) * 1000.0,
            )

    def init_first_buffer(self, blocks, adapter_block_idx=None):
        with torch_device_module.stream(self.init_stream):
            with get_wan_runtuner().gpu_region(
                "offload.initial_copy", {"block_index": 0}
            ):
                if hasattr(self, "cpu_buffers"):
                    if self.offload_granularity == "block":
                        self.cuda_buffers[0].load_state_dict(self.cpu_buffers[0].state_dict(), 0, adapter_block_idx)
                    else:
                        self.cuda_buffers[0].load_state_dict(self.cpu_buffers[0][0].state_dict(), 0, adapter_block_idx)
                else:
                    if self.offload_granularity == "block":
                        self.cuda_buffers[0].load_state_dict(blocks[0].state_dict(), 0, adapter_block_idx)
                    else:
                        self.cuda_buffers[0].load_state_dict(blocks[0].compute_phases[0].state_dict(), 0, adapter_block_idx)
        self._sync()
        self.need_init_first_buffer = False

    def prefetch_weights(self, block_idx, blocks, adapter_block_idx=None):
        with torch_device_module.stream(self.cuda_load_stream):
            with get_wan_runtuner().gpu_region(
                "offload.prefetch_copy", {"block_index": int(block_idx)}
            ):
                if hasattr(self, "cpu_buffers"):
                    self.cuda_buffers[1].load_state_dict(self.cpu_buffers[0].state_dict(), block_idx, adapter_block_idx)
                else:
                    self.cuda_buffers[1].load_state_dict(blocks[block_idx].state_dict(), block_idx, adapter_block_idx)

    def prefetch_phase(self, block_idx, phase_idx, blocks, adapter_block_idx=None):
        with torch_device_module.stream(self.cuda_load_stream):
            if hasattr(self, "cpu_buffers"):
                self.cuda_buffers[phase_idx].load_state_dict(self.cpu_buffers[0][phase_idx].state_dict(), block_idx, adapter_block_idx)
            else:
                self.cuda_buffers[phase_idx].load_state_dict(blocks[block_idx].compute_phases[phase_idx].state_dict(), block_idx, adapter_block_idx)

    def swap_blocks(self):
        tuner = get_wan_runtuner()
        if AI_DEVICE == "xpu":
            started = time.perf_counter() if tuner.active else None
            torch_device_module.synchronize()
            if started is not None:
                tuner.record_cpu_ms(
                    "offload.device_wait",
                    (time.perf_counter() - started) * 1000.0,
                )
        else:
            started = time.perf_counter() if tuner.active else None
            self.cuda_load_stream.synchronize()
            if started is not None:
                tuner.record_cpu_ms(
                    "offload.load_wait",
                    (time.perf_counter() - started) * 1000.0,
                )
            started = time.perf_counter() if tuner.active else None
            self.compute_stream.synchronize()
            if started is not None:
                tuner.record_cpu_ms(
                    "offload.compute_wait",
                    (time.perf_counter() - started) * 1000.0,
                )
        self.cuda_buffers[0], self.cuda_buffers[1] = (
            self.cuda_buffers[1],
            self.cuda_buffers[0],
        )

    def swap_phases(self):
        tuner = get_wan_runtuner()
        started = time.perf_counter() if tuner.active else None
        if AI_DEVICE == "xpu":
            torch_device_module.synchronize()
        else:
            self.cuda_load_stream.synchronize()
            self.compute_stream.synchronize()
        if started is not None:
            tuner.record_cpu_ms(
                "offload.phase_wait",
                (time.perf_counter() - started) * 1000.0,
            )

    @ExcludedProfilingContext("🔥 warm_up_cpu_buffers")
    def warm_up_cpu_buffers(self, blocks_num):
        logger.info("🔥 Warming up cpu buffers...")
        for i in tqdm(range(blocks_num)):
            for phase in self.cpu_buffers[0]:
                phase.load_state_dict_from_disk(i, None)
            for phase in self.cpu_buffers[1]:
                phase.load_state_dict_from_disk(i, None)

        for phase in self.cpu_buffers[0]:
            phase.load_state_dict_from_disk(0, None)
        for phase in self.cpu_buffers[1]:
            phase.load_state_dict_from_disk(1, None)
        logger.info("✅ CPU buffers warm-up completed.")

    def init_lazy_load(self, num_workers=6):
        self.lazy_load = True
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_futures = []
        self.prefetch_block_idx = -1

    def start_prefetch_block(self, block_idx, adapter_block_idx=None):
        self.prefetch_block_idx = block_idx
        self.prefetch_futures = []
        if self.offload_granularity == "block":
            future = self.executor.submit(self.cpu_buffers[1].load_state_dict_from_disk, block_idx, adapter_block_idx)
            self.prefetch_futures.append(future)
        else:
            for phase in self.cpu_buffers[1]:
                future = self.executor.submit(phase.load_state_dict_from_disk, block_idx, adapter_block_idx)
                self.prefetch_futures.append(future)

    def swap_cpu_buffers(self):
        # import time
        # wait_start = time.time()
        # already_done = all(f.done() for f in self.prefetch_futures)
        for f in self.prefetch_futures:
            f.result()
        # wait_time = time.time() - wait_start
        # logger.debug(f"[Prefetch] block {self.prefetch_block_idx}: wait={wait_time:.3f}s, already_done={already_done}")
        self.cpu_buffers = [self.cpu_buffers[1], self.cpu_buffers[0]]

    def __del__(self):
        if hasattr(self, "executor") and self.executor is not None:
            for f in self.prefetch_futures:
                if not f.done():
                    f.result()
            self.executor.shutdown(wait=False)
            self.executor = None
            logger.debug("ThreadPoolExecutor shut down successfully.")
