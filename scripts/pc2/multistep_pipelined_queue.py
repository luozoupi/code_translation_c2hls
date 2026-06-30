"""SQLite job queue for pipelined multistep (codegen / synth workers).

Reuses the flash queue implementation; multistep job ``phase`` values include
``phase_b`` and each of ``DEFAULT_OPT_STEPS``.
"""

from __future__ import annotations

from flash_pipelined_queue import FlashPipelinedQueue, PipelinedJob

MultistepPipelinedQueue = FlashPipelinedQueue

__all__ = ["MultistepPipelinedQueue", "PipelinedJob"]
