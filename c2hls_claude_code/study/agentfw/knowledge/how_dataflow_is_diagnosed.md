dfdiag: does an HLS design really overlap load / compute / store?

Evidence, never the DATAFLOW label alone:
  * source     : DATAFLOW pragmas and where they sit (in a loop?), explicit [2] buffers / ping-pong indexing,
                 hls::stream_of_blocks, hls::stream channels.
  * csynth log : regions Vitis actually turned into dataflow ("Applying dataflow ... extracted N process"),
                 channels it built (PIPO = ping-pong memories, FIFO), FIFO depth warnings.
  * csynth rpt : per-module latency and interval, per-instance (process) latency, loop tables (trip, II).
For every dataflow region R with processes P_i:
  S = sum(lat P_i), M = max(lat P_i), L = lat R, II = interval R.
  intra-invocation overlap  = (S - L) / (S - M)          1.0 = all processes run together, 0.0 = one after another
  across iterations (R inside a loop of N trips, loop latency T):
      pipelined if T ~= N*II + (L - II)   (next iteration's region starts while the previous one drains)
      serial    if T ~= N*L
Classes (design level):
  NO_DATAFLOW, INEFFECTIVE (pragma present, no real overlap), TASK_CONCURRENCY (processes overlap, single
  invocation over the whole problem, e.g. stream pipelines), PINGPONG (region in a tile loop, iterations overlap
  through PIPO/double buffers), and an overlap grade: PARTIAL or FULL (latency within 10% of the bottleneck stage's
  total work, plus fill). Work outside every dataflow region (e.g. a serial preload) is reported as serial overhead.

Usage: dfdiag.py <work_dir> [--src kernel.cpp] [--json out.json]
  work_dir = the Vitis synthesis dir holding hls_proj/sol1/syn/report and logs/hls_run_tcl.log
